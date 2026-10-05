"""Unit tests for ciii module."""

import pytest

from ciii import (
    classify_estimate,
    quadrant_split,
    score_ciii,
    score_g1,
    score_g2,
    score_g3,
)


def test_score_g1_unknown_returns_zero():
    assert score_g1("完全不認識") == 0
    assert score_g1("完全明白") == 3
    assert score_g1("garbage") == 0


def test_score_g2_unknown_returns_zero():
    assert score_g2("不清楚") == 0
    assert score_g2("$10,000 或以上") == 2


def test_score_g3_unknown_returns_zero():
    assert score_g3("不清楚") == 0
    assert score_g3("$120,000–$200,000（高回報）") == 4
    assert score_g3("garbage") == 0


def test_score_ciii_sum():
    assert score_ciii("完全明白", "$10,000 或以上", "$120,000–$200,000（高回報）") == 3 + 2 + 4


def test_score_ciii_handles_unknown_values():
    assert score_ciii("unknown", "unknown", "unknown") == 0


def test_classify_estimate_unknown():
    r = classify_estimate(None)
    assert r["key"] == "unknown"
    r = classify_estimate(0)
    assert r["key"] == "unknown"
    r = classify_estimate(-100)
    assert r["key"] == "unknown"


def test_classify_estimate_correct_range():
    r = classify_estimate(150_000)
    assert r["key"] == "correct"


def test_classify_estimate_near_range():
    r = classify_estimate(80_000)
    assert r["key"] == "near"


def test_classify_estimate_serious_low():
    r = classify_estimate(30_000)
    assert r["key"] == "serious"


def test_classify_estimate_over():
    r = classify_estimate(300_000)
    assert r["key"] == "over"


def test_classify_estimate_custom_thresholds():
    r = classify_estimate(500, correct_min=100, correct_max=1000)
    assert r["key"] == "correct"


def test_quadrant_split_basic():
    ciii = [1, 1, 5, 5, 3, 3]
    will = [2, 5, 2, 5, 2, 5]
    out = quadrant_split(ciii, will, ciii_threshold=2, willingness_threshold=4)
    assert out["low_ciii_low_willingness"] == 1
    assert out["low_ciii_high_willingness"] == 1
    assert out["high_ciii_low_willingness"] == 2
    assert out["high_ciii_high_willingness"] == 2


def test_quadrant_split_length_mismatch_raises():
    with pytest.raises(ValueError):
        quadrant_split([1, 2], [3])