import streamlit as st, pandas as pd, numpy as np, plotly.express as px
import plotly.graph_objects as go
from ml_utils import (
    train_classifiers, plot_confusion, plot_roc,
    forecast_prophet, forecast_arima,
    kmeans_segmentation, build_optimisation
)
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.metrics import mean_absolute_error
import base64, io, time

st.set_page_config(page_title="Hospitality & Aviation Analytics", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("synthetic_hosp_aviation_dataset.csv", parse_dates=["date"])

df = load_data()

# Sidebar Filters
st.sidebar.header("Global Filters")
date_min, date_max = df["date"].min(), df["date"].max()
date_range = st.sidebar.date_input("Date range", (date_min, date_max))
mask = (df["date"] >= pd.to_datetime(date_range[0])) & (df["date"] <= pd.to_datetime(date_range[1]))
for col in ["property_type","location","loyalty_tier","segment","booking_channel","risk_level"]:
    opts = st.sidebar.multiselect(col, sorted(df[col].unique()), default=sorted(df[col].unique()))
    mask &= df[col].isin(opts)
filtered_df = df[mask]

# Page Nav
tabs = [
    "Data Visualisation", "Forecasting & Dynamic Pricing", "Recommendation & Sentiment",
    "Classification", "Clustering & Segmentation", "Association & A/B",
    "Optimisation & Real-Time Ops", "Regression & Risk/FDM"
]
page = st.sidebar.radio("Select Tab", tabs)
st.title(page)

def download_button(obj, filename, label):
    buffer = io.BytesIO()
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(buffer, index=False)
    else:
        buffer.write(obj.encode())
    b64 = base64.b64encode(buffer.getvalue()).decode()
    href = f'<a href="data:text/csv;base64,{b64}" download="{filename}">{label}</a>'
    st.markdown(href, unsafe_allow_html=True)

# 1 Data Visualisation (few charts for brevity)
if page == "Data Visualisation":
    st.subheader("Revenue over Time")
    st.plotly_chart(px.line(filtered_df, x="date", y="revenue", color="property_type"), use_container_width=True)
    st.caption("Insight: revenue trend by property type.")

    st.subheader("Utilisation Histogram")
    st.plotly_chart(px.histogram(filtered_df, x="resource_utilization", nbins=30), use_container_width=True)

    st.subheader("Delay Heatmap")
    heat = filtered_df.pivot_table(values="delay_minutes", index="location", columns="risk_level", aggfunc="mean")
    st.plotly_chart(px.imshow(heat, text_auto=True))

# 2 Forecasting
elif page == "Forecasting & Dynamic Pricing":
    target = st.selectbox("Target", ["units_sold","revenue","dynamic_price"])
    model_type = st.radio("Model", ["Prophet","ARIMA"])
    horizon = st.slider("Forecast horizon (days)", 7, 60, 30)
    if model_type=="Prophet":
        forecast = forecast_prophet(filtered_df, target, horizon)
    else:
        forecast = forecast_arima(filtered_df, target, horizon)
    st.plotly_chart(px.line(forecast, x="ds", y="yhat", title=f"{model_type} forecast → {target}"), use_container_width=True)
    st.caption("Confidence bands shown where available.")

# 3 Recommendation & Sentiment
elif page == "Recommendation & Sentiment":
    st.subheader("Sentiment Histogram")
    st.plotly_chart(px.histogram(filtered_df, x="sentiment_score", nbins=40), use_container_width=True)

# 4 Classification (simplified to show patch)
elif page == "Classification":
    target = st.selectbox("Target label", ["promo_response"])
    X = pd.get_dummies(filtered_df.drop(columns=[target]), drop_first=True)
    y = filtered_df[target]
    res = train_classifiers(X,y)
    metric_df = pd.DataFrame({k:{"Train":v["train_acc"],"Test":v["test_acc"]} for k,v in res.items()}).T
    st.dataframe(metric_df)

# other pages omitted for brevity