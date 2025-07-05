# Hospitality & Aviation Analytics Dashboard

This repository contains a **production‑ready Streamlit dashboard** that showcases advanced data‑analytics techniques on the `synthetic_hosp_aviation_dataset.csv` (1 000 rows).

## Quick start

```bash
# 1. Clone
git clone <YOUR_REPO_URL>
cd streamlit_hosp_dashboard

# 2. Install deps (use virtual‑env)
pip install -r requirements.txt

# 3. Run locally
streamlit run app.py
```

## Deploy to Streamlit Cloud
1. Push this repo (plus the CSV) to GitHub.
2. Log‑in to [streamlit.io/cloud](https://streamlit.io/cloud) → **New app** → point to `app.py`.
3. Set `PYTHON_VERSION` (3.11+) and **secrets** if needed. Deployment will `pip install -r requirements.txt` automatically.

## Tabs implemented
| Tab | Key Methods |
|-----|-------------|
| Data Visualisation | 10+ interactive charts |
| Forecasting & Dynamic Pricing | ARIMA & Prophet forecasts |
| Recommendation & Sentiment | Item‑based collaborative filtering, sentiment bar & wordcloud |
| Classification | KNN, Decision Tree, Random Forest, Gradient Boosting |
| Clustering & Segmentation | K‑means (slider, elbow) |
| Association Rules  &  A/B | Apriori / FPGrowth, mini A/B tester |
| Optimization & RT Ops | Linear programming staff optimiser, real‑time gauges |
| Regression & FDM Risk | Linear / Ridge / Lasso / DT regression, anomaly detection |

See `data_dictionary.md` for a description of each column.