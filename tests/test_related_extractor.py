import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from src.related.extractor import (
    RelationExtractor,
    _SYSTEM_PROMPT,
    _truncate_value_chain,
    normalize_name,
    resolve_ticker,
)
from src.related.models import ReportSections


def test_system_prompt_has_direction_and_merger_rules():
    """Regression guard: the prompt must keep the direction (parent vs
    subsidiary) and merged-entity-exclusion directives that prevent the
    known extraction errors (e.g. labeling a parent holdco as a subsidiary,
    or keeping a company that was absorbed by merger)."""
    # 방향 규약: 모회사·지배기업은 Affiliate, 본 기업이 지배하는 것만 Subsidiary
    assert "모회사" in _SYSTEM_PROMPT
    assert "지배기업" in _SYSTEM_PROMPT
    assert "방향" in _SYSTEM_PROMPT
    # 합병 소멸 회사 제외
    assert "흡수합병" in _SYSTEM_PROMPT
    assert ("소멸" in _SYSTEM_PROMPT) or ("해산" in _SYSTEM_PROMPT)
    # 밸류체인(고객·공급처) 추출 — 계열사만 뽑히던 문제 방지
    assert "매출처" in _SYSTEM_PROMPT
    assert "Customer" in _SYSTEM_PROMPT and "Supplier" in _SYSTEM_PROMPT
    assert "익명" in _SYSTEM_PROMPT


def test_truncate_preserves_value_chain_lines_when_cut():
    """사업의 내용은 길어서 앞부분만 남기면 뒤쪽 '주요 매출처/원재료' 표가
    잘려 고객·공급망 관계가 영영 추출되지 않는다. 잘릴 때 해당 줄은 보존해야 한다."""
    filler = "회사의 일반적인 사업 개요 설명입니다.\n" * 200
    tail = "주요 매출처: 삼성전자, SK하이닉스\n원재료 매입처: A사"
    text = filler + tail

    out = _truncate_value_chain(text, max_chars=500)

    assert len(out) <= 500 + 40  # truncation marker 여유
    assert "주요 매출처" in out
    assert "삼성전자" in out
    assert "원재료 매입처" in out


def test_truncate_value_chain_returns_text_unchanged_when_short():
    text = "짧은 본문. 주요 매출처: 삼성전자"
    assert _truncate_value_chain(text, max_chars=1000) == text


def test_normalize_name_strips_corporate_suffixes():
    assert normalize_name("(주)삼성전자") == "삼성전자"
    assert normalize_name("삼성전자(주)") == "삼성전자"
    assert normalize_name("주식회사 한미반도체") == "한미반도체"
    assert normalize_name("SK하이닉스주식회사") == "SK하이닉스"


def test_resolve_ticker_exact_match():
    name_to_ticker = {"삼성전자": "005930", "SK하이닉스": "000660"}
    assert resolve_ticker("삼성전자", name_to_ticker) == "005930"
    assert resolve_ticker("(주)삼성전자", name_to_ticker) == "005930"


def test_resolve_ticker_unknown_returns_none():
    name_to_ticker = {"삼성전자": "005930"}
    assert resolve_ticker("ASML", name_to_ticker) is None


@pytest.mark.asyncio
async def test_extract_calls_openai_with_json_mode():
    sections = ReportSections(
        corp_code="00126380", rcept_no="20250318",
        business_content="당사는 메모리 반도체를 제조한다. 주요 고객사는 Apple입니다.",
        affiliates="", related_party="", related_party_notes="",
    )

    mock_message = MagicMock()
    mock_message.content = json.dumps({
        "edges": [
            {"name": "Apple", "ticker": None, "relation": "Customer",
             "evidence": "주요 고객사는 Apple입니다."},
        ]
    })
    mock_choice = MagicMock(); mock_choice.message = mock_message
    mock_resp = MagicMock(); mock_resp.choices = [mock_choice]

    mock_openai = MagicMock()
    mock_openai.chat.completions.create = AsyncMock(return_value=mock_resp)

    with patch("openai.AsyncOpenAI", return_value=mock_openai):
        extractor = RelationExtractor(api_key="test-key", model="gpt-5-nano")
        edges = await extractor.extract(
            target_name="삼성전자", target_ticker="005930",
            sections=sections, name_to_ticker={"삼성전자": "005930"},
        )

    assert len(edges) == 1
    assert edges[0].name == "Apple"
    assert edges[0].relation == "Customer"
    call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
    assert call_kwargs.get("response_format") == {"type": "json_object"}


@pytest.mark.asyncio
async def test_extract_resolves_unknown_tickers_from_name_map():
    """If GPT returns ticker=None but name is in the map, resolve it."""
    sections = ReportSections(
        corp_code="00", rcept_no="0",
        business_content="x", affiliates="", related_party="", related_party_notes="",
    )

    mock_message = MagicMock()
    mock_message.content = json.dumps({
        "edges": [
            {"name": "SK하이닉스", "ticker": None, "relation": "Customer",
             "evidence": "HBM 후공정"},
        ]
    })
    mock_choice = MagicMock(); mock_choice.message = mock_message
    mock_resp = MagicMock(); mock_resp.choices = [mock_choice]

    mock_openai = MagicMock()
    mock_openai.chat.completions.create = AsyncMock(return_value=mock_resp)

    with patch("openai.AsyncOpenAI", return_value=mock_openai):
        extractor = RelationExtractor(api_key="key", model="gpt-5-nano")
        edges = await extractor.extract(
            target_name="한미반도체", target_ticker="042700",
            sections=sections,
            name_to_ticker={"SK하이닉스": "000660"},
        )

    assert edges[0].ticker == "000660"
