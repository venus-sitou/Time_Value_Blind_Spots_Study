# ⏱️ Time-Value Blind Spots — Quantitative Analysis Scripts

> Privacy-preserving Python scripts that demonstrate **how to analyse**
> compound-interest-intuition questionnaire data using a diagnostic-index
> framework. The framework, *not* any specific empirical result, is what
> this repository shares.

These scripts were developed alongside an exploratory student research
project on time-value reasoning. The repository contains only the
**computation pipeline** — no original participant data is included
or implied.

The companion **interactive diagnostic tool** lives in a separate
repository: <https://github.com/your-username/ciii-diagnostic-app>.

---

## ⚠️ Privacy and scope

- This repository contains **only generic, reusable analysis code**.
- It contains **no original survey responses**, no questionnaire exports,
  no school names, and no personal or class information.
- Any numerical examples in the code are either derived from clearly-synthetic
  test fixtures or are obvious formula outputs (e.g. `MOP $144,026` for
  `MOP $300 × 18y @ 8%`).
- The research project's exact empirical results are intentionally not
  reproduced here. To run the pipeline on real data, supply your own
  appropriately anonymised CSV at the repo root.

## 🔬 What this code demonstrates

The scripts showcase four core capabilities useful for a quant-finance /
data-science portfolio:

| Capability | Where in code |
|---|---|
| **Compound-interest math primitives** | `finance_math.future_value_monthly_payment` |
| **Quantitative scoring of open-ended responses** | `ciii.score_ciii` |
| **Robust statistics with HC3 standard errors** | `scripts.run_ols_hc3` |
| **Multiple-testing correction (BH-FDR)** | `scripts.run_fdr` |

The diagnostic index logic itself (what counts as a "time-value blind spot")
is implemented in the **CIII Diagnostic Tool**, not here.

---

## 📁 Project Structure

```
time-value-blind-spots-study/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
│
├── finance_math.py            # Compound interest primitives
├── ciii.py                     # Diagnostic-index scoring logic
├── stats_utils.py              # FDR + HC3 utilities
├── plotting.py                 # Reusable charts
│
├── analyze_from_csv.py         # Main entry point
├── generate_mock_data.py       # Synthetic CSV generator
├── quick_analysis.py           # Minimal example
├── enhanced_analysis_v3.py     # Extended example
│
├── data/
│   ├── README.md
│   └── synthetic_compound_intuition_responses.csv   # synthetic demo data
│
├── notebooks/
│   └── demo_analysis.ipynb     # Walkthrough notebook
│
└── tests/
    └── test_*.py               # pytest unit tests
```

---

## 🚀 Quick Start

```bash
git clone <this-repo>
cd time-value-blind-spots-study

pip install -r requirements.txt

# 1. Generate synthetic data (or supply your own CSV at repo root)
python generate_mock_data.py

# 2. Run the main analysis on the synthetic CSV
python analyze_from_csv.py
```

The script writes a structured summary of statistics to
`outputs/analysis_results.json`.

---

## 📦 Reproducibility and unit tests

```bash
pytest tests/ -v
```

---

## 🛠️ Methodology (illustrative, not research-claiming)

| Step | Output |
|---|---|
| Load CSV (synthetic or supplied) | DataFrame |
| Compute compound future value | per-row column |
| Compute diagnostic index score | per-row column |
| Aggregate descriptive statistics | table |
| Run Spearman / OLS / HC3 | table |
| Apply Benjamini-Hochberg FDR | adjusted p-values |
| Save plots to `outputs/figures/` | PNG files |

This is a *demonstration pipeline*. It does not produce or claim any
specific empirical result.

---

## 🔒 Privacy

- The pipeline never logs or transmits individual respondent data.
- The pipeline can run entirely on **synthetic data** without
  modification.
- The companion diagnostic app is also privacy-first (no accounts, no
  telemetry, no API calls).

---

## 🤝 Contributing

Issues and pull requests welcome for educational improvements:

- Better statistical documentation
- New scenario examples
- Accessibility
- Additional i18n

We do not welcome pull requests that:

- Add tracking, telemetry, analytics, or accounts
- Add data persistence beyond a local JSON dump
- Convert this into a financial product promotion tool

---

## 📄 License

MIT License — see [LICENSE](LICENSE).

---

## 📚 Suggested Reading

- [Time Value of Money](https://en.wikipedia.org/wiki/Time_value_of_money)
- [Compound Interest](https://en.wikipedia.org/wiki/Compound_interest)
- [Benjamini–Hochberg FDR](https://en.wikipedia.org/wiki/False_discovery_rate#Benjamini%E2%80%93Hochberg_procedure)
- [HC3 Robust Standard Errors (MacKinnon, White 1985)](https://en.wikipedia.org/wiki/Heteroscedasticity-consistent_standard_errors)
- [OECD Financial Literacy Framework](https://www.oecd.org/financial/education/)

---

## ✍️ Author note

This repository was developed as part of a personal exploration of
time-value reasoning in financial education. It does **not** represent
any institutional research finding. The empirical work that
accompanied this code was exploratory, with a small convenience sample,
and is intentionally **not** part of this public repository.