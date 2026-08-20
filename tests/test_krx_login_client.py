# tests/test_krx_login_client.py
from unittest.mock import MagicMock

import pytest

from src.krx_login_client import KrxLoginClient, KrxBlockedError, _is_block_page, _BLD


_BLOCK_HTML = """<html>
<head>
  <title>에러페이지 - 한국거래소 | Data Marketplace</title>
</head>
<body><div class="ip-block-page"><div class="ip-block-login-header"></div></div></body>
</html>"""


def test_is_block_page_detects_krx_block():
    assert _is_block_page(_BLOCK_HTML) is True
    # 다른 문구의 KRX 에러페이지도 감지
    assert _is_block_page('<title>에러페이지 - 한국거래소 | Data Marketplace</title>') is True


def test_is_block_page_false_for_normal_bodies():
    assert _is_block_page('{"OutBlock_1": [{"a": 1}]}') is False
    assert _is_block_page("LOGOUT") is False
    assert _is_block_page("") is False


def test_post_raises_and_marks_blocked_on_block_page():
    """차단 페이지를 받으면 KrxBlockedError를 올리고 blocked 플래그를 세운다."""
    c = KrxLoginClient(krx_id="x", krx_pw="y")
    c._session_initialized = True  # skip network init
    c._logged_in = True
    resp = MagicMock()
    resp.text = _BLOCK_HTML
    resp.status_code = 200
    c._session = MagicMock()
    c._session.post = MagicMock(return_value=resp)

    with pytest.raises(KrxBlockedError):
        c._post("some/bld", {})
    assert c._blocked is True


def test_post_short_circuits_when_already_blocked():
    """이미 차단 감지된 상태면 네트워크 요청 없이 즉시 중단한다(추가 요청으로 차단 악화 방지)."""
    c = KrxLoginClient(krx_id="x", krx_pw="y")
    c._blocked = True
    c._session = MagicMock()
    c._session.post = MagicMock(side_effect=AssertionError("네트워크 요청이 발생하면 안 됨"))

    with pytest.raises(KrxBlockedError):
        c._post("some/bld", {})
    c._session.post.assert_not_called()


def test_get_all_market_ohlcv_maps_kosdaq_global_and_drops_konex():
    """MKT_NM 매핑: KOSPI/KOSDAQ 그대로, KOSDAQ GLOBAL은 KOSDAQ로 합치고
    KONEX는 버린다 — 저장소 20260819 KOSPI 942·KOSDAQ 1,821(1,771+50)건과
    정확히 일치함을 실측으로 확인한 매핑."""
    c = KrxLoginClient(krx_id="x", krx_pw="y")
    rows = [
        {"ISU_SRT_CD": "005930", "MKT_NM": "KOSPI", "TDD_HGPRC": "273,000",
         "TDD_CLSPRC": "271,000", "CMPPREVDD_PRC": "23,500"},
        {"ISU_SRT_CD": "035720", "MKT_NM": "KOSDAQ", "TDD_HGPRC": "50,000",
         "TDD_CLSPRC": "49,000", "CMPPREVDD_PRC": "-1,000"},
        {"ISU_SRT_CD": "999999", "MKT_NM": "KOSDAQ GLOBAL", "TDD_HGPRC": "1,000",
         "TDD_CLSPRC": "900", "CMPPREVDD_PRC": "0"},
        {"ISU_SRT_CD": "111111", "MKT_NM": "KONEX", "TDD_HGPRC": "1,000",
         "TDD_CLSPRC": "900", "CMPPREVDD_PRC": "0"},
    ]
    c._post = MagicMock(return_value=rows)

    df = c.get_all_market_ohlcv("20260820")

    assert set(df.index) == {"005930", "035720", "999999"}
    assert df.loc["005930", "시장"] == "KOSPI"
    assert df.loc["035720", "시장"] == "KOSDAQ"
    assert df.loc["999999", "시장"] == "KOSDAQ"  # KOSDAQ GLOBAL -> KOSDAQ
    assert df.loc["005930", "고가"] == 273000
    assert df.loc["005930", "종가"] == 271000
    assert df.loc["005930", "전일대비"] == 23500


def test_get_all_market_ohlcv_requests_mkt_id_all():
    c = KrxLoginClient(krx_id="x", krx_pw="y")
    c._post = MagicMock(return_value=[])
    c.get_all_market_ohlcv("20260820")
    c._post.assert_called_once_with(_BLD["ohlcv_by_ticker"], {"mktId": "ALL", "trdDd": "20260820"})


def test_get_all_market_ohlcv_empty_when_no_rows():
    c = KrxLoginClient(krx_id="x", krx_pw="y")
    c._post = MagicMock(return_value=[])
    df = c.get_all_market_ohlcv("20260820")
    assert df.empty
