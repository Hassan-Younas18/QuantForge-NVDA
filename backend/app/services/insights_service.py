"""
Narrative generation over already-computed results (run_summary.json,
eda.json). Pure presentation logic — no modeling happens here.
"""
from __future__ import annotations

from . import registry


def build_insights() -> dict:
    summary = registry.load_run_summary()
    eda = registry.load_eda()
    if summary is None:
        return {
            "why_selected": "No trained model yet. Run a training job to see model insights.",
            "trend_summary": "No data analysed yet.",
            "confidence": "unknown",
        }

    best_name = summary["best_model"]
    metrics = summary["metrics"]
    best = metrics[best_name]
    naive = metrics.get("naive_rw")

    why_parts = [
        f"The {best_name.upper()} model was selected because it achieved the lowest "
        f"validation RMSE (${best['rmse']:.2f}) among the candidates "
        f"({', '.join(k for k in metrics if k != 'naive_rw')})."
    ]
    if naive:
        rmse_gap_pct = (naive["rmse"] - best["rmse"]) / naive["rmse"] * 100
        if rmse_gap_pct > 0:
            why_parts.append(
                f"It edges out the naive random-walk baseline by {rmse_gap_pct:.2f}% on RMSE."
            )
        else:
            why_parts.append(
                f"Note: the naive random-walk baseline actually beats it by "
                f"{-rmse_gap_pct:.2f}% on RMSE — a common finding for daily price "
                f"levels, where 'tomorrow = today' is a brutally strong baseline."
            )
        why_parts.append(
            f"Directional accuracy is {best.get('dir_acc', 0):.1f}% "
            f"(50% is a coin flip), which is the more honest measure of skill."
        )

    trend_summary = "No EDA available."
    if eda:
        trend_summary = (
            f"Over the analysed window ({eda['start']} to {eda['end']}), the stock "
            f"returned {eda['total_return_pct']:.1f}% total ({eda['cagr_pct']:.1f}% CAGR) "
            f"with {eda['annual_vol_pct']:.1f}% annualised volatility. "
            f"Daily return skew is {eda['skew']:.2f} and kurtosis {eda['kurtosis']:.2f}, "
            f"indicating {'fat tails / occasional sharp moves' if eda['kurtosis'] > 3 else 'roughly normal tail behaviour'}."
        )

    dir_acc = best.get("dir_acc", 50.0)
    if dir_acc >= 56:
        confidence = "moderate"
    elif dir_acc >= 51:
        confidence = "low"
    else:
        confidence = "very low"

    return {
        "why_selected": " ".join(why_parts),
        "trend_summary": trend_summary,
        "confidence": confidence,
        "confidence_note": (
            "Confidence reflects directional accuracy on held-out test data, not "
            "a guarantee. Forecast intervals capture model uncertainty (MC-Dropout), "
            "not full market risk — see the project README's Honest Limitations."
        ),
    }
