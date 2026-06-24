from __future__ import annotations

from pydantic import BaseModel


class InsightsResponse(BaseModel):
    why_selected: str
    trend_summary: str
    confidence: str
    confidence_note: str | None = None
