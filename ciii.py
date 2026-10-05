"""
ciii.py — illustrative diagnostic-index scoring.

This module demonstrates how a survey-response-based "compound interest
intuition index" (CIII) could be constructed and scored. It is
**purely illustrative** — the actual scoring thresholds used in the
original research project are not reproduced here.

Educational purpose:
    Show how a small-N quantitative index can be assembled from
    ordinal/Likert-style survey responses, normalised to a 0-10 scale,
    and used to split a sample into analytic subgroups.
"""

from typing import Iterable, Sequence


# === Illustrative encoding schemes ===
# These mappings are illustrative. The original project may have used
# different encodings; values here are chosen for clarity and to make
# the pipeline runnable on synthetic data.

G1_SCORE_MAP = {
    "完全不認識": 0,
    "聽過但不清楚細節": 1,
    "稍微認識": 2,
    "完全明白": 3,
}

# Illustrative 30-year-savings encoding
G2_SCORE_MAP = {
    "不清楚": 0,
    "1000": 0,
    "$2,000～$3,000": 1,
    "$5,000～$10,000": 2,
    "$10,000 或以上": 2,
}

# Illustrative 18-year-annuity encoding
G3_SCORE_MAP = {
    "不清楚": 0,
    "$30,000 以下（等於總投入）": 1,
    "$30,000–$60,000（少少回報）": 2,
    "$60,000–$120,000（中等回報）": 3,
    "$120,000–$200,000（高回報）": 4,
}


def score_g1(g1_value) -> int:
    """Score G1 self-reported compound-interest familiarity."""
    return G1_SCORE_MAP.get(g1_value, 0)


def score_g2(g2_value) -> int:
    """Score G2 30-year savings estimation."""
    return G2_SCORE_MAP.get(str(g2_value), 0)


def score_g3(g3_value) -> int:
    """Score G3 18-year compound future-value estimation."""
    return G3_SCORE_MAP.get(g3_value, 0)


def score_ciii(g1_value, g2_value, g3_value) -> int:
    """Compute the compound interest intuition index (0-10).

    Sum of three sub-scores. Each sub-score is bounded by its encoding
    max (3, 2, 4 respectively), so the max total is 9. We treat the
    max as a soft cap and report the sum.

    Note: this is one way to construct such an index; it is not
    psychometrically validated.
    """
    return score_g1(g1_value) + score_g2(g2_value) + score_g3(g3_value)


def classify_estimate(
    estimate: float | None,
    *,
    correct_min: float = 120_000,
    correct_max: float = 200_000,
    near_min: float = 60_000,
) -> dict:
    """Classify a single compound-value estimate into a teaching category.

    These thresholds are calibrated to a specific teaching scenario
    (MOP $300/mo × 18y × 8% → MOP $144,026). They should be revisited
    when the underlying scenario changes.

    Returns dict with keys: key, label, emoji, message.
    """
    if estimate is None or estimate <= 0:
        return {
            "key": "unknown", "label": "不知道", "emoji": "❓",
            "message": "No value provided — the student did not enter an estimate.",
        }
    if correct_min <= estimate <= correct_max:
        return {
            "key": "correct", "label": "正確量級", "emoji": "✅",
            "message": "Estimate is within the correct range.",
        }
    if near_min <= estimate < correct_min:
        return {
            "key": "near", "label": "接近但低估", "emoji": "⚠️",
            "message": "Estimate is below the correct range but above principal.",
        }
    if estimate < near_min:
        return {
            "key": "serious", "label": "嚴重低估", "emoji": "🚨",
            "message": "Estimate is at or below principal — suggests linear thinking.",
        }
    return {
        "key": "over", "label": "高估", "emoji": "⬆️",
        "message": "Estimate is above the correct range.",
    }


def quadrant_split(
    ciii_values: Sequence[int],
    willingness_values: Sequence[int],
    *,
    ciii_threshold: int = 2,
    willingness_threshold: int = 4,
) -> dict:
    """Two-by-two quadrant split.

    Splits by median-or-cutoff CIII and willingness-to-take-finance-class.

    Returns dict with 4 quadrant counts.
    """
    n = len(ciii_values)
    if n != len(willingness_values):
        raise ValueError("Lengths must match")
    out = {"low_ciii_low_willingness": 0, "low_ciii_high_willingness": 0,
           "high_ciii_low_willingness": 0, "high_ciii_high_willingness": 0}
    for c, w in zip(ciii_values, willingness_values):
        lc = c <= ciii_threshold
        hw = w >= willingness_threshold
        if lc and not hw: out["low_ciii_low_willingness"] += 1
        elif lc and hw: out["low_ciii_high_willingness"] += 1
        elif not lc and not hw: out["high_ciii_low_willingness"] += 1
        else: out["high_ciii_high_willingness"] += 1
    return out


__all__ = [
    "G1_SCORE_MAP", "G2_SCORE_MAP", "G3_SCORE_MAP",
    "score_g1", "score_g2", "score_g3", "score_ciii",
    "classify_estimate", "quadrant_split",
]