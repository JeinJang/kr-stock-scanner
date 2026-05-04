from pydantic import BaseModel


class CorpInfo(BaseModel):
    """Listed corporation master data from Open DART."""

    corp_code: str        # 8-digit DART internal code
    ticker: str           # 6-digit stock code
    name: str
    market: str           # "KOSPI" | "KOSDAQ"


class FinancialStatement(BaseModel):
    """A single account value from a financial report."""

    corp_code: str
    year: int
    quarter: int          # 0 = annual report, 1/2/3 = quarter reports
    account: str          # e.g. "매출액", "영업이익", "당기순이익"
    value: float
