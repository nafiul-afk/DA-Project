# ReadmitRisk — Code

Source code for the ReadmitRisk Hospital Readmission Risk & Resource Allocation Platform, developed for Data Analytics Laboratory, Summer 2026, United International University.

This is intentionally a **code-only** repository. It excludes the source dataset, trained models, generated tables/figures, notebook outputs, PDFs, and presentation files.

## Setup

Use Python 3.12 from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock.txt
```

Place the original UCI/Kaggle dataset files here:

```text
data/diabetic_data.csv
data/IDS_mapping.csv
```

Then build all outputs and launch the dashboard:

```bash
python run_project.py --phase all --force
streamlit run app.py
```

To run the test suite:

```bash
pytest -q
```

## Repository layout

- `readmitrisk/` — preprocessing, EDA, modeling, sequence forecasting, allocation, clustering, and reporting modules
- `scripts/` — notebook, dashboard capture, and submission-validation utilities
- `tests/` — automated tests
- `app.py` — Streamlit dashboard
- `run_project.py` — reproducible pipeline runner
