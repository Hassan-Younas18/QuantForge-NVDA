"""
Data acquisition from Yahoo Finance.

Robust to the quirks of modern ``yfinance``:
  * single-ticker downloads sometimes return MultiIndex columns,
  * ``auto_adjust`` changes whether 'Close' is split/dividend adjusted.
We request *unadjusted* OHLCV plus a separate 'Adj Close' so both are
available, then cache to parquet to avoid hammering the API on re-runs.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from ..config import CACHE_DIR, DataConfig

logger = logging.getLogger("nvda")

EXPECTED_COLS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def _cache_path(cfg: DataConfig) -> Path:
    return CACHE_DIR / f"{cfg.ticker}_{cfg.period_years}y_{cfg.interval}.parquet"


def _is_cache_fresh(path: Path, max_age_days: int) -> bool:
    if not path.exists():
        return False
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    return age < timedelta(days=max_age_days)


def _normalise_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Flatten possible MultiIndex columns and guarantee EXPECTED_COLS."""
    if isinstance(df.columns, pd.MultiIndex):
        # Either (field, ticker) or (ticker, field) ordering can occur.
        lvl0 = set(df.columns.get_level_values(0))
        if ticker in lvl0:
            df = df.xs(ticker, axis=1, level=0)
        else:
            df = df.xs(ticker, axis=1, level=1)
    df = df.rename(columns={c: c.title() for c in df.columns})
    if "Adj Close" not in df.columns and "Adj_Close" in df.columns:
        df = df.rename(columns={"Adj_Close": "Adj Close"})
    if "Adj Close" not in df.columns:          # auto_adjust collapsed it
        df["Adj Close"] = df["Close"]
    return df[[c for c in EXPECTED_COLS if c in df.columns]]


def download_data(cfg: DataConfig, retries: int = 3) -> pd.DataFrame:
    """
    Download (or load from cache) OHLCV history for ``cfg.ticker``.

    Returns a DatetimeIndex-ed DataFrame with EXPECTED_COLS, ascending dates.
    """
    cache = _cache_path(cfg)
    if cfg.use_cache and _is_cache_fresh(cache, cfg.cache_max_age_days):
        logger.info("Loading cached data -> %s", cache)
        return pd.read_parquet(cache)

    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "yfinance is required for live download. `pip install yfinance`."
        ) from exc

    end = datetime.now()
    start = end - timedelta(days=int(cfg.period_years * 365.25) + 5)
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            logger.info("Downloading %s [%s -> %s] (attempt %d)",
                        cfg.ticker, start.date(), end.date(), attempt)
            raw = yf.download(
                cfg.ticker, start=start, end=end, interval=cfg.interval,
                auto_adjust=False, progress=False, threads=False,
            )
            if raw is None or raw.empty:
                raise ValueError("Empty frame returned from Yahoo Finance.")
            df = _normalise_columns(raw, cfg.ticker)
            df = df.sort_index()
            df.index.name = "Date"
            if cfg.use_cache:
                # Write to a temp file then atomically replace the cache path —
                # concurrent requests (e.g. the API serving history + indicators
                # for the same ticker at once) would otherwise race on writing
                # the same file and corrupt it.
                tmp = cache.with_suffix(f".{os.getpid()}.tmp")
                df.to_parquet(tmp)
                os.replace(tmp, cache)
                logger.info("Cached -> %s", cache)
            logger.info("Downloaded %d rows (%s -> %s)",
                        len(df), df.index.min().date(), df.index.max().date())
            return df
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            logger.warning("Download failed: %s", exc)
            time.sleep(2 * attempt)
    raise RuntimeError(f"Could not download {cfg.ticker} after {retries} tries: "
                       f"{last_err}")


def get_company_info(ticker: str) -> dict:
    """Best-effort company metadata (name, sector, market cap, 52w range)."""
    try:
        import yfinance as yf

        info = yf.Ticker(ticker).get_info()
        return {
            "ticker": ticker,
            "short_name": info.get("shortName") or info.get("longName") or ticker,
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "market_cap": info.get("marketCap"),
            "currency": info.get("currency", "USD"),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
            "previous_close": info.get("previousClose"),
            "logo_url": f"https://logo.clearbit.com/{info.get('website', '').replace('https://', '').replace('http://', '').rstrip('/')}" if info.get("website") else None,
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not fetch company info for %s: %s", ticker, exc)
        return {"ticker": ticker, "short_name": ticker}


def get_earnings_dates(ticker: str, limit: int = 40) -> pd.DatetimeIndex:
    """Best-effort fetch of historical/upcoming earnings dates (optional feature)."""
    try:
        import yfinance as yf

        ed = yf.Ticker(ticker).get_earnings_dates(limit=limit)
        if ed is not None and not ed.empty:
            return pd.DatetimeIndex(ed.index).tz_localize(None).normalize()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not fetch earnings dates: %s", exc)
    return pd.DatetimeIndex([])
