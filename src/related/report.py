from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import networkx as nx
import plotly
from jinja2 import Environment, FileSystemLoader

from src.related.db import StoredEdge


_REL_COLORS = {
    "Supplier": "#1e8449",
    "Customer": "#1864c0",
    "Competitor": "#c0392b",
    "Affiliate": "#6a1ec0",
    "Subsidiary": "#d35400",
}


class ReportGenerator:
    """Generates an interactive HTML report for a single ticker."""

    def __init__(self):
        template_dir = Path(__file__).parent / "templates"
        env = Environment(loader=FileSystemLoader(str(template_dir)))
        env.filters["tojson"] = lambda v: json.dumps(v, ensure_ascii=False)
        self._env = env

    def generate(
        self,
        root_ticker: str,
        graph: nx.DiGraph,
        edges: list[StoredEdge],
        name_map: dict[str, str],
        market_map: dict[str, str],
        scores: dict[str, dict],
        depth: int,
        rcept_no: str,
        as_of_date: str,
        output_dir: str,
    ) -> str:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"related-{root_ticker}-{as_of_date}.html")

        pos = nx.spring_layout(graph, seed=42) if len(graph) else {}

        nodes_payload = []
        for node in graph.nodes:
            x, y = pos.get(node, (0, 0))
            attrs = graph.nodes[node]
            if node.startswith("_unlisted_"):
                label = attrs.get("display_name", node[10:])
                color = "#aaaaaa"
                hover = f"{label} (비상장/외국)"
                size = 14
            else:
                label = name_map.get(node, node)
                if node == root_ticker:
                    color = "#e74c3c"
                else:
                    color = "#3498db"
                sc = scores.get(node, {})
                grade = sc.get("grade", "")
                cats = ", ".join(sc.get("categories", []))
                hover = f"{label} ({node})<br>{market_map.get(node, '')}<br>{grade} {cats}".strip()
                base_size = 12
                if sc.get("total_score") is not None:
                    base_size = 8 + sc["total_score"] / 10
                size = 22 if node == root_ticker else base_size
            nodes_payload.append({
                "id": node, "x": float(x), "y": float(y),
                "label": label, "color": color, "size": size, "hover": hover,
            })

        edges_payload = []
        for u, v, attrs in graph.edges(data=True):
            x0, y0 = pos.get(u, (0, 0))
            x1, y1 = pos.get(v, (0, 0))
            edges_payload.append({
                "x0": float(x0), "y0": float(y0),
                "x1": float(x1), "y1": float(y1),
                "relation": attrs.get("relation", ""),
                "evidence": attrs.get("evidence", ""),
            })

        graph_data = {"nodes": nodes_payload, "edges": edges_payload}

        root_edges = [e for e in edges if e.source_ticker == root_ticker]
        grouped: dict[str, list[dict]] = defaultdict(list)
        for e in root_edges:
            sc = scores.get(e.target_ticker or "", {}) if e.target_ticker else {}
            grouped[e.relation].append({
                "name": e.target_name,
                "ticker": e.target_ticker,
                "grade": sc.get("grade", ""),
                "evidence": e.evidence,
            })

        plotly_js = plotly.offline.get_plotlyjs()
        template = self._env.get_template("report.html")
        html = template.render(
            root_ticker=root_ticker,
            root_name=name_map.get(root_ticker, root_ticker),
            root_market=market_map.get(root_ticker, ""),
            depth=depth,
            node_count=len(graph.nodes),
            edge_count=len(graph.edges),
            grouped=dict(grouped),
            rcept_no=rcept_no,
            as_of_date=as_of_date,
            graph_data=graph_data,
            plotly_js=plotly_js,
        )
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        return output_path
