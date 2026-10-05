"""
analyze_from_csv.py — main entry point for the analysis pipeline.

This script reads a CSV with the expected questionnaire column layout,
runs the diagnostic-index and statistics pipeline, and writes a
structured summary to `outputs/analysis_results.json`.

It is designed to:
- Run on **synthetic** data (for portfolio demonstration).
- Run on **real** data (if the user supplies their own CSV at the repo root).
  Note: real survey data is gitignored and never committed.

It does NOT:
- Make any claims about specific empirical results.
- Reference the original research project, competition, or institution.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))

from finance_math import future_value_monthly_payment, opportunity_cost
from ciii import score_ciii


EXPECTED_COLUMNS = {
    "G1": "Compound-interest familiarity (Likert)",
    "G2": "30-year savings estimation",
    "G3": "18-year compound future-value estimation",
    "G6": "Product awareness (multi-select, comma-separated)",
    "G9": "Willingness to take finance class (Likert)",
}


def find_data_csv() -> Path | None:
    candidates = [
        Path("questionnaire_responses_v21.csv"),
        Path("data/synthetic_compound_intuition_responses.csv"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        print(f"Warning: CSV is missing expected columns: {missing_cols}")
    return df


def compute_ciii_column(df: pd.DataFrame) -> pd.Series:
    return df.apply(
        lambda r: score_ciii(r.get("G1"), r.get("G2"), r.get("G3")), axis=1
    )


def classify_g3(series: pd.Series) -> dict:
    strict = (series == "$120,000–$200,000（高回報）").sum()
    lenient = series.isin({
        "$120,000–$200,000（高回報）",
        "$60,000–$120,000（中等回報）",
    }).sum()
    unknown = (series == "不清楚").sum()
    n = len(series)
    return {
        "n": n,
        "strict_correct": {"n": int(strict), "pct": strict / n * 100 if n else 0},
        "lenient_correct": {"n": int(lenient), "pct": lenient / n * 100 if n else 0},
        "unknown": {"n": int(unknown), "pct": unknown / n * 100 if n else 0},
    }


def spearman_ciii_willingness(df: pd.DataFrame, ciii: pd.Series) -> dict:
    will = pd.to_numeric(df["G9"], errors="coerce")
    mask = will.notna()
    if mask.sum() < 3:
        return {"rho": None, "p": None, "n": int(mask.sum())}
    rho, p = stats.spearmanr(ciii[mask], will[mask])
    return {"rho": float(rho), "p": float(p), "n": int(mask.sum())}


def main():
    print("=" * 60)
    print("CIII × TVB-20 Analysis Pipeline")
    print("=" * 60)

    csv_path = find_data_csv()
    if csv_path is None:
        print("No CSV found.")
        print("   Run `python generate_mock_data.py` to create a synthetic dataset,")
        print("   or place your own `questionnaire_responses_v21.csv` at the repo root.")
        sys.exit(1)

    print(f"Loading: {csv_path}")
    df = load_data(csv_path)
    print(f"   N = {len(df)} rows")

    ciii = compute_ciii_column(df)
    g3_stats = classify_g3(df["G3"])
    oc = opportunity_cost(300, 0.08, 18)
    sp = spearman_ciii_willingness(df, ciii)

    results = {
        "input_file": str(csv_path),
        "n": len(df),
        "g3_distribution": g3_stats,
        "ciii": {
            "mean": float(ciii.mean()) if len(ciii) else None,
            "median": float(ciii.median()) if len(ciii) else None,
            "sd": float(ciii.std()) if len(ciii) else None,
            "max": int(ciii.max()) if len(ciii) else None,
        },
        "opportunity_cost_demo_MOP300_18y_8pct": oc,
        "spearman_ciii_x_G9": sp,
        "notes": (
            "This is a demonstration pipeline. Numerical results depend entirely "
            "on the input CSV. Replace the synthetic CSV with your own "
            "(appropriately anonymised) data to obtain real results."
        ),
    }

    out_path = Path("outputs/analysis_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Wrote: {out_path}")

    print("\n=== Summary ===")
    print(f"N = {results['n']}")
    print(f"G3 strict correct: {results['g3_distribution']['strict_correct']['n']} "
          f"({results['g3_distribution']['strict_correct']['pct']:.1f}%)")
    print(f"G3 don't know:    {results['g3_distribution']['unknown']['n']} "
          f"({results['g3_distribution']['unknown']['pct']:.1f}%)")
    if results['ciii']['mean'] is not None:
        print(f"CIII mean / median / sd / max: "
              f"{results['ciii']['mean']:.2f} / "
              f"{results['ciii']['median']:.1f} / "
              f"{results['ciii']['sd']:.2f} / "
              f"{results['ciii']['max']}")
    print(f"Spearman CIII x G9: rho = {results['spearman_ciii_x_G9']['rho']:.3f}, "
          f"p = {results['spearman_ciii_x_G9']['p']:.4f}")
    print(f"Opportunity cost demo (MOP 300/mo x 18y x 8%): FV = MOP {oc['future_value']:,.0f}")


if __name__ == "__main__":
    main()
