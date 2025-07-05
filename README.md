# Hospitality & Aviation Analytics Dashboard

This repository contains a **production‑ready Streamlit dashboard** powered by advanced analytics on `synthetic_hosp_aviation_dataset.csv` (1 000 rows).

> **Python runtime pinned to 3.11.9** (see `runtime.txt`) with compatible SciPy/Statsmodels to avoid `_lazywhere` import errors on Streamlit Cloud.

## Quick start

```bash
git clone <YOUR_REPO_URL>
cd streamlit_hosp_dashboard
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Cloud
Just push this repo (including the CSV) and create a new app pointing to `app.py`. Streamlit Cloud reads `runtime.txt`, builds a 3.11 image, installs the pinned wheels, and launches.

All analytic tabs, models, and visualisations specified in the assignment are implemented. See `data_dictionary.md` for column details.