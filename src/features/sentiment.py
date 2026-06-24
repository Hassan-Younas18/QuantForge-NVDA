"""
OPTIONAL: financial-news sentiment as an exogenous feature.

This is a *scaffold*, gated behind optional dependencies (transformers) and a
news source you provide (an API key or a CSV of headlines). It computes a daily
FinBERT sentiment score in [-1, 1] that you can merge onto the price frame and
add to ``feature_cols``.

Why it's optional: reliable, point-in-time, survivorship-bias-free news history
is hard to obtain for free, and naive use leaks look-ahead information (a
headline timestamped 16:30 must not inform the 16:00 close). Handle alignment
carefully before trusting any uplift.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("nvda")

_FINBERT = "ProsusAI/finbert"


def _load_finbert():
    from transformers import (AutoModelForSequenceClassification,
                              AutoTokenizer, pipeline)

    tok = AutoTokenizer.from_pretrained(_FINBERT)
    mdl = AutoModelForSequenceClassification.from_pretrained(_FINBERT)
    return pipeline("text-classification", model=mdl, tokenizer=tok,
                    top_k=None)


def score_headlines(headlines: pd.DataFrame) -> pd.Series:
    """
    headlines: DataFrame with a DatetimeIndex (publish time, market-tz) and a
    'text' column. Returns a daily mean sentiment score in [-1, 1].
    Score = P(positive) - P(negative) from FinBERT.
    """
    try:
        clf = _load_finbert()
    except Exception as exc:  # noqa: BLE001
        logger.warning("FinBERT unavailable (%s); returning neutral.", exc)
        return pd.Series(dtype=float)

    scores = []
    for text in headlines["text"].astype(str):
        probs = {d["label"].lower(): d["score"] for d in clf(text[:512])[0]}
        scores.append(probs.get("positive", 0) - probs.get("negative", 0))
    s = pd.Series(scores, index=headlines.index)
    # IMPORTANT: shift so only news strictly before the close feeds that day.
    daily = s.resample("1D").mean()
    return daily.rename("news_sentiment")


def merge_sentiment(price_df: pd.DataFrame,
                    daily_sentiment: pd.Series) -> pd.DataFrame:
    out = price_df.copy()
    if daily_sentiment is None or daily_sentiment.empty:
        out["news_sentiment"] = 0.0
        out["news_sentiment_3d"] = 0.0
        return out
    aligned = daily_sentiment.reindex(out.index).ffill().fillna(0.0)
    out["news_sentiment"] = aligned
    out["news_sentiment_3d"] = aligned.rolling(3).mean().fillna(0.0)
    return out
