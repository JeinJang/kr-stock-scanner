from __future__ import annotations

import json
import re

import openai
from loguru import logger

from src.related.models import ExtractedEdge, ReportSections


_CORPORATE_SUFFIXES = [
    "(주)", "(주식회사)", "주식회사", "(유)", "유한회사",
    "Co., Ltd.", "Co.,Ltd.", "Inc.", "Ltd.", "Corp.",
]


def normalize_name(name: str) -> str:
    """Strip whitespace and corporate suffixes for matching."""
    out = name.strip()
    out = re.sub(r"\(주식회사\)|\(주\)|\(유\)", "", out)
    for suffix in ["주식회사", "유한회사"]:
        if out.startswith(suffix):
            out = out[len(suffix):]
        if out.endswith(suffix):
            out = out[: -len(suffix)]
    for suffix in ["Co., Ltd.", "Co.,Ltd.", "Inc.", "Ltd.", "Corp."]:
        if out.endswith(suffix):
            out = out[: -len(suffix)].strip(",.")
    return out.strip()


def resolve_ticker(name: str, name_to_ticker: dict[str, str]) -> str | None:
    """Resolve a company name to a Korean ticker, or None."""
    if name in name_to_ticker:
        return name_to_ticker[name]
    norm = normalize_name(name)
    if norm in name_to_ticker:
        return name_to_ticker[norm]
    norm_map = {normalize_name(k): v for k, v in name_to_ticker.items()}
    return norm_map.get(norm)


_SYSTEM_PROMPT = """당신은 한국 기업의 사업보고서를 분석하는 전문가입니다.
주어진 텍스트에서 본 기업과 다른 기업 간의 관계를 추출해 JSON으로 출력하세요.

관계 타입은 모두 "본 기업 기준"으로, 상대 기업(target)이 본 기업에 대해 갖는 관계입니다:
- Supplier: 공급업체 (본 기업이 매입하는 거래처)
- Customer: 고객사 (본 기업이 매출을 일으키는 거래처)
- Competitor: 경쟁사
- Affiliate: 같은 그룹의 계열사. **본 기업을 지배하는 모회사·지배기업·최대주주도 여기에 포함**합니다(이들은 본 기업의 자회사가 아니므로 절대 Subsidiary로 분류하지 마세요).
- Subsidiary: **본 기업이 지분을 보유·지배하는 자회사·종속회사만** 해당합니다. 본 기업이 지배당하는(피지배) 관계는 Subsidiary가 아닙니다.

규칙:
- 본 기업 자체나 자기 자신은 포함하지 마세요.
- **방향 주의(가장 흔한 오류):** '출자회사/피출자회사', '지배기업/종속기업' 표에서는 방향을 반대로 적기 쉽습니다. 본 기업이 상대를 지배하면 Subsidiary, 상대가 본 기업을 지배하면(=모회사·중간지주) Affiliate로 분류하세요.
- **소멸 회사 제외:** 흡수합병되어 소멸했거나 청산·해산된 회사, 합병으로 존속법인에 통합돼 더 이상 별도로 존재하지 않는 회사는 관계에서 제외하세요.
- **밸류체인(고객·공급처)을 반드시 찾으세요:** 계열사·자회사만 나열하고 끝내면 안 됩니다. '사업의 내용'의 주요 매출처·수요처·납품처는 **Customer**, 원재료 매입처·공급처·외주처는 **Supplier**로 추출하세요. 소재·부품·장비 기업은 전방 대기업 고객이 핵심 정보입니다.
- **익명 거래처도 기록:** 보고서가 'A사', '국내 대형 반도체업체'처럼 익명 처리했더라도 그대로 name에 적고 ticker는 null로 두세요. 누락하지 마세요.
- 각 항목에 evidence(원문에서 인용한 짧은 근거 문장)를 반드시 포함하세요.
- ticker는 6자리 한국 종목코드. 모르거나 비상장/외국기업이면 null.
- 출력 스키마: {"edges": [{"name": str, "ticker": str|null, "relation": str, "evidence": str}]}
"""


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated]"


# 고객사·공급처 정보가 담긴 줄을 식별하는 키워드. '사업의 내용'은 매우 길어
# 앞부분만 남기면 뒤쪽의 주요 매출처·원재료 표가 잘려 밸류체인(Customer/
# Supplier) 관계가 아예 추출되지 않습니다.
_VALUE_CHAIN_KEYWORDS = (
    "매출처", "고객", "수요처", "납품처", "거래처",
    "원재료", "매입처", "공급처", "외주",
)


def _truncate_value_chain(text: str, max_chars: int) -> str:
    """Truncate long text but keep lines that name customers/suppliers.

    앞부분(70%)을 그대로 두고, 잘려나갈 뒷부분에서 밸류체인 키워드가 포함된
    줄만 골라 남은 예산만큼 덧붙입니다.
    """
    if len(text) <= max_chars:
        return text
    head_budget = int(max_chars * 0.7)
    head = text[:head_budget]
    kept = [
        line for line in text[head_budget:].split("\n")
        if any(kw in line for kw in _VALUE_CHAIN_KEYWORDS)
    ]
    if not kept:
        return head + "\n... [truncated]"
    tail = "\n".join(kept)[: max_chars - head_budget]
    return head + "\n... [truncated, 밸류체인 관련 발췌]\n" + tail


class RelationExtractor:
    """Extract company relationships from report sections using GPT."""

    def __init__(self, api_key: str, model: str = "gpt-5-nano",
                 max_chars_per_section: int = 24000):
        self._client = openai.AsyncOpenAI(api_key=api_key)
        self._model = model
        self._max_chars = max_chars_per_section

    async def extract(
        self,
        target_name: str,
        target_ticker: str,
        sections: ReportSections,
        name_to_ticker: dict[str, str],
    ) -> list[ExtractedEdge]:
        body = (
            f"대상 기업: {target_name} ({target_ticker})\n\n"
            f"=== 사업의 내용 ===\n{_truncate_value_chain(sections.business_content, self._max_chars)}\n\n"
            f"=== 계열회사 등 ===\n{_truncate(sections.affiliates, self._max_chars)}\n\n"
            f"=== 대주주 등과의 거래 ===\n{_truncate(sections.related_party, self._max_chars)}\n\n"
            f"=== 특수관계자 거래 ===\n{_truncate(sections.related_party_notes, self._max_chars)}\n"
        )

        resp = await self._client.chat.completions.create(
            model=self._model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": body},
            ],
        )
        content = resp.choices[0].message.content or "{}"
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed for {target_ticker}: {e}")
            return []

        edges_raw = data.get("edges", [])
        result: list[ExtractedEdge] = []
        for item in edges_raw:
            name = (item.get("name") or "").strip()
            if not name:
                continue
            relation = item.get("relation")
            if relation not in ("Supplier", "Customer", "Competitor",
                                "Affiliate", "Subsidiary"):
                continue
            ticker = item.get("ticker")
            if not ticker:
                ticker = resolve_ticker(name, name_to_ticker)
            else:
                if ticker not in set(name_to_ticker.values()):
                    ticker = resolve_ticker(name, name_to_ticker)
            evidence = (item.get("evidence") or "").strip()
            result.append(ExtractedEdge(
                name=name, ticker=ticker, relation=relation, evidence=evidence,
            ))
        return result
