from __future__ import annotations

from pydantic import BaseModel


class StockInfo(BaseModel):
    ticker: str
    short_name: str
    sector: str | None = None
    industry: str | None = None
    market_cap: float | None = None
    currency: str = "USD"
    fifty_two_week_low: float | None = None
    fifty_two_week_high: float | None = None
    previous_close: float | None = None
    logo_url: str | None = None
    last_close: float
    change: float
    change_pct: float
    last_date: str
    volume: float | None = None


class OhlcvBar(BaseModel):
    Date: str
    Open: float | None = None
    High: float | None = None
    Low: float | None = None
    Close: float | None = None
    model_config = {"extra": "allow"}


class IndicatorBar(BaseModel):
    Date: str
    Close: float | None = None
    model_config = {"extra": "allow"}
