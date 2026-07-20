import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from src.related.extractor import (
    RelationExtractor,
    _SYSTEM_PROMPT_STRUCTURE,
    _SYSTEM_PROMPT_VALUE_CHAIN,
    _truncate_value_chain,
    normalize_name,
    resolve_ticker,
)
from src.related.models import ReportSections


def test_structure_prompt_has_direction_and_merger_rules():
    """계열 패스 프롬프트는 방향 규약(모회사↔자회사)과 합병 소멸 제외를 유지해야 한다."""
    assert "모회사" in _SYSTEM_PROMPT_STRUCTURE
    assert "지배기업" in _SYSTEM_PROMPT_STRUCTURE
    assert "방향" in _SYSTEM_PROMPT_STRUCTURE
    assert "흡수합병" in _SYSTEM_PROMPT_STRUCTURE
    assert ("소멸" in _SYSTEM_PROMPT_STRUCTURE) or ("해산" in _SYSTEM_PROMPT_STRUCTURE)


def test_value_chain_prompt_has_customer_supplier_rules():
    """밸류체인 패스 프롬프트는 고객·공급처 추출과 익명 거래처 기록을 지시해야 한다."""
    assert "매출처" in _SYSTEM_PROMPT_VALUE_CHAIN
    assert "Customer" in _SYSTEM_PROMPT_VALUE_CHAIN
    assert "Supplier" in _SYSTEM_PROMPT_VALUE_CHAIN
    assert "익명" in _SYSTEM_PROMPT_VALUE_CHAIN


def _mock_resp(payload: dict):
    msg = MagicMock(); msg.content = json.dumps(payload)
    choice = MagicMock(); choice.message = msg
    resp = MagicMock(); resp.choices = [choice]
    return resp


@pytest.mark.asyncio
async def test_extract_runs_two_passes_and_merges_both_categories():
    """한 번의 호출로는 계열/밸류체인 중 한쪽만 나오는 문제가 있어 패스를 분리한다.
    구조(계열) 패스와 밸류체인 패스를 각각 호출하고 결과를 합쳐야 한다."""
    sections = ReportSections(
        corp_code="00", rcept_no="0",
        business_content="주요 매출처: A사 38%",
        affiliates="계열회사: 자회사케이",
        related_party="", related_party_notes="",
    )
    structure = _mock_resp({"edges": [
        {"name": "자회사케이", "ticker": None, "relation": "Subsidiary", "evidence": "계열회사"},
    ]})
    value_chain = _mock_resp({"edges": [
        {"name": "A사", "ticker": None, "relation": "Customer", "evidence": "주요 매출처 38%"},
    ]})

    mock_openai = MagicMock()
    mock_openai.chat.completions.create = AsyncMock(side_effect=[structure, value_chain])

    with patch("openai.AsyncOpenAI", return_value=mock_openai):
        extractor = RelationExtractor(api_key="k")
        edges = await extractor.extract(
            target_name="심텍", target_ticker="222800",
            sections=sections, name_to_ticker={},
        )

    assert mock_openai.chat.completions.create.await_count == 2
    relations = {e.relation for e in edges}
    assert "Subsidiary" in relations, "계열 패스 결과가 누락되면 안 됨"
    assert "Customer" in relations, "밸류체인 패스 결과가 누락되면 안 됨"


@pytest.mark.asyncio
async def test_extract_dedupes_same_company_across_passes():
    """두 패스가 같은 회사를 반환해도 한 번만 남아야 한다(중복 방지)."""
    sections = ReportSections(
        corp_code="00", rcept_no="0", business_content="x",
        affiliates="y", related_party="", related_party_notes="",
    )
    dup_a = _mock_resp({"edges": [
        {"name": "(주)동일기업", "ticker": None, "relation": "Affiliate", "evidence": "계열"},
    ]})
    dup_b = _mock_resp({"edges": [
        {"name": "동일기업", "ticker": None, "relation": "Customer", "evidence": "매출처"},
    ]})

    mock_openai = MagicMock()
    mock_openai.chat.completions.create = AsyncMock(side_effect=[dup_a, dup_b])

    with patch("openai.AsyncOpenAI", return_value=mock_openai):
        extractor = RelationExtractor(api_key="k")
        edges = await extractor.extract(
            target_name="심텍", target_ticker="222800",
            sections=sections, name_to_ticker={},
        )

    assert len(edges) == 1, f"중복 제거 실패: {[e.name for e in edges]}"


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
