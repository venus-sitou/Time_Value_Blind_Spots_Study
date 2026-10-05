# Synthetic Data README

This directory contains **synthetic demonstration data only**.

The CSV file in this directory is **not** real participant data. It was
generated programmatically to demonstrate the analysis pipeline in the
parent repository. It is intended for educational and reproducibility
demonstration purposes.

## File: synthetic_compound_intuition_responses.csv

Synthetic survey responses that follow the *column structure* of the
original questionnaire (A1, A2, A3, A4, A5, A6, A7, A8, B1–B20, C1–C12,
D1–D12, E1–G10). However, all values are randomly generated within
realistic bounds.

## Important

- **Do not interpret results from the synthetic data as research findings.**
- The synthetic data has no relationship to any real-world population.
- It exists solely to demonstrate that the analysis scripts run end-to-end
  without modification.

## Regenerating

To regenerate the synthetic data, run:

```bash
python scripts/generate_mock_data.py
```

(This script is intentionally public because it produces no real data.)

## If you have your own (real) data

- Place it at the repo root as `questionnaire_responses_v21.csv`.
- It will be ignored by `.gitignore` and never pushed.
- The analysis pipeline reads it automatically.