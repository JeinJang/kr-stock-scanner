# tests/test_krx_login_client.py
from unittest.mock import MagicMock

import pytest

from src.krx_login_client import KrxLoginClient, KrxBlockedError, _is_block_page


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
