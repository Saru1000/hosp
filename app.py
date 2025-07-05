import streamlit as st, pandas as pd, numpy as np, plotly.express as px, plotly.graph_objects as go
from ml_utils import train_classifiers, plot_confusion, plot_roc, forecast_arima, forecast_hwes, kmeans_segmentation, optimise_staff
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.metrics import mean_absolute_error
import base64, io

st.set_page_config(page_title="Hosp & Aviation Dashboard", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("synthetic_hosp_aviation_dataset.csv", parse_dates=["date"])

df = load_data()

# Sidebar filters
st.sidebar.header("Filters")
date_min, date_max = df["date"].min(), df["date"].max()
date_range = st.sidebar.date_input("Date range",(date_min,date_max))
mask = (df["date"]>=pd.to_datetime(date_range[0]))&(df["date"]<=pd.to_datetime(date_range[1]))
for col in ["property_type","location","loyalty_tier","segment","booking_channel","risk_level"]:
    opts = st.sidebar.multiselect(col, sorted(df[col].dropna().unique()), default=list(df[col].dropna().unique()))
    mask &= df[col].isin(opts)
fdf = df[mask]

page = st.sidebar.radio("Page", ["Viz","Forecast","Classify"])
st.title(page)

def dl_button(obj, fname, label):
    buf=io.BytesIO()
    if isinstance(obj,pd.DataFrame): obj.to_csv(buf,index=False)
    else: buf.write(obj.encode())
    b64=base64.b64encode(buf.getvalue()).decode()
    st.markdown(f'<a href="data:text/csv;base64,{b64}" download="{fname}">{label}</a>',unsafe_allow_html=True)

if page=="Viz":
    st.plotly_chart(px.line(fdf,x="date",y="revenue",color="property_type"), use_container_width=True)
elif page=="Forecast":
    target = st.selectbox("Target", ["units_sold","revenue"])
    ts = fdf.set_index("date")[target].asfreq("D").interpolate(limit_direction='both')
    model = st.radio("Model", ["ARIMA","HoltWinters"])
    steps = st.slider("Horizon", 7,60,30)
    if model=="ARIMA":
        fc = forecast_arima(ts, steps)
    else:
        fc = forecast_hwes(ts, steps)
    st.plotly_chart(px.line(fc,x="ds",y="yhat"), use_container_width=True)
elif page=="Classify":
    target = "promo_response"
    X = pd.get_dummies(fdf.drop(columns=[target]), drop_first=True)
    y = fdf[target]
    res = train_classifiers(X,y)
    st.dataframe({k:{'Test':v['test_acc']} for k,v in res.items()})