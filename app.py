import streamlit as st
import pandas as pd, numpy as np, plotly.express as px, plotly.graph_objects as go
from ml_utils import (
    train_classifiers, plot_confusion, plot_roc,
    forecast_prophet, kmeans_segmentation, build_optimisation
)
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tempfile import NamedTemporaryFile
import base64, io, time, datetime, itertools

st.set_page_config(page_title="Hospitality & Aviation Analytics", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("synthetic_hosp_aviation_dataset.csv", parse_dates=["date"])

df = load_data()

# --- SIDEBAR ---
st.sidebar.header("Global Filters")
date_min, date_max = df["date"].min(), df["date"].max()
date_range = st.sidebar.date_input("Date range", (date_min, date_max))
mask = (df["date"] >= pd.to_datetime(date_range[0])) & (df["date"] <= pd.to_datetime(date_range[1]))

for col in ["property_type","location","loyalty_tier","segment","booking_channel","risk_level"]:
    vals = st.sidebar.multiselect(f"Filter {col}", sorted(df[col].dropna().unique()), default=sorted(df[col].dropna().unique()))
    mask &= df[col].isin(vals)
filtered_df = df[mask]

# --- PAGE NAV ---
tabs = [
    "Data Visualisation", "Forecasting & Dynamic Pricing", "Recommendation & Sentiment",
    "Classification", "Clustering & Segmentation", "Association & A/B",
    "Optimisation & Real-Time Ops", "Regression & Risk/FDM"
]
page = st.sidebar.radio("Select Tab", tabs)

st.title(page)

# Helper to download DataFrame
def download_button(object_to_download, download_filename, button_text):
    buffer = io.BytesIO()
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download.to_csv(buffer, index=False)
    else:
        buffer.write(object_to_download.encode())
    b64 = base64.b64encode(buffer.getvalue()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{button_text}</a>'
    st.markdown(href, unsafe_allow_html=True)

# ---------- 1. Data Visualisation ----------
if page == "Data Visualisation":
    with st.expander("How to read this page"):
        st.write("Interactive charts driven by the global filters above.")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Revenue", f"${filtered_df['revenue'].sum():,.0f}")
    col2.metric("Avg. Utilization", f"{filtered_df['resource_utilization'].mean():.2%}")
    col3.metric("Mean Sentiment", f"{filtered_df['sentiment_score'].mean():.2f}")

    st.subheader("Revenue over time")
    fig = px.line(filtered_df, x="date", y="revenue", color="property_type")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Insight: Revenue seasonality differs between property types.")

    st.subheader("Utilization distribution")
    st.plotly_chart(px.histogram(filtered_df, x="resource_utilization", nbins=30), use_container_width=True)

    st.subheader("Heat‑map: Avg delay by location vs risk_level")
    heat = filtered_df.pivot_table(values="delay_minutes", index="location", columns="risk_level", aggfunc="mean")
    st.plotly_chart(px.imshow(heat, text_auto=True, color_continuous_scale='Reds'))

    st.subheader("Pair‑plot (sample)")
    sample = filtered_df.sample(min(500,len(filtered_df)))
    st.plotly_chart(px.scatter_matrix(sample, dimensions=["dynamic_price","units_sold","revenue","delay_minutes"], color="property_type"))

    st.subheader("Booking channel share")
    st.plotly_chart(px.pie(filtered_df, names="booking_channel", values="units_sold", hole=0.45))

    st.subheader("Top 10 insights")
    top_box = st.text_area("Insights",
        "\n".join([
            f"1. Peak revenue of ${filtered_df.groupby('date')['revenue'].sum().max():,.0f} occurs on {filtered_df.groupby('date')['revenue'].sum().idxmax().date()}",
            f"2. {filtered_df['risk_level'].value_counts(normalize=True).idxmax()} risk dominates ({filtered_df['risk_level'].value_counts(normalize=True).max():.0%})",
            "... (add more)"
        ]),
        height=200
    )

# ---------- 2. Forecasting & Dynamic Pricing ----------
elif page == "Forecasting & Dynamic Pricing":
    with st.expander("How to read this page"):
        st.write("Time‑series forecasts for units_sold and revenue. Compare ARIMA vs Prophet.")
    ts_col = st.selectbox("Select target", ["units_sold", "revenue", "dynamic_price"])
    periods = st.slider("Forecast horizon (days)", 7, 60, 30)
    st.write("Training Prophet...")
    model, fc = forecast_prophet(filtered_df, ts_col, periods)
    fig = px.line(fc, x="ds", y=["yhat", "yhat_lower", "yhat_upper"])
    st.plotly_chart(fig, use_container_width=True)
    mae = mean_absolute_error(fc["y"].dropna(), fc["yhat"].dropna())
    st.metric("MAE", f"{mae:,.2f}")
    st.caption("Dynamic pricing recommendation: increase/decrease based on forecast vs competitor_price.")

# ---------- 3. Recommendation & Sentiment ----------
elif page == "Recommendation & Sentiment":
    with st.expander("How to read this page"):
        st.write("Item‑based collaborative filtering on customer × ancillary_purchase.")
    # Sentiment bar
    st.subheader("Sentiment distribution")
    st.plotly_chart(px.histogram(filtered_df, x="sentiment_score", nbins=40), use_container_width=True)
    # Simple recommender: probability to buy ancillary
    pivot = filtered_df.pivot_table(index="customer_id", columns="ancillary_purchase", values="revenue", aggfunc="count").fillna(0)
    # item-based similarity
    if 1 in pivot.columns:
        from sklearn.metrics.pairwise import cosine_similarity
        sim = cosine_similarity(pivot[[1]])
        st.write("Item-similarity calculated; (demo) recommending ancillary to top customers.")
        top_cust = pivot.sum(axis=1).nlargest(10).index
        rec_df = pd.DataFrame({"customer_id": top_cust, "recommend_ancillary": True})
        st.dataframe(rec_df)
        download_button(rec_df, "recommendations.csv", "Download recommendations")

# ---------- 4. Classification ----------
elif page == "Classification":
    st.subheader("Model training")
    target_col = st.selectbox("Target label", ["promo_response","maintenance_flag","ancillary_purchase"])
    X = pd.get_dummies(filtered_df.drop(columns=[target_col]), drop_first=True)
    y = filtered_df[target_col]
    st.write(f"Training on {X.shape[0]} records, {X.shape[1]} features")
    results = train_classifiers(X, y)
    metric_df = pd.DataFrame({k:{'Train Acc':v['train_acc'],'Test Acc':v['test_acc'],
                                 'Precision':v['precision'],'Recall':v['recall'],'F1':v['f1']} for k,v in results.items()}).T
    st.dataframe(metric_df)

    algo = st.selectbox("Show confusion matrix for", list(results.keys()))
    st.plotly_chart(plot_confusion(results[algo]["y_test"], results[algo]["y_pred"], labels=y.unique()))

    st.subheader("ROC curves")
    st.plotly_chart(plot_roc(results), use_container_width=True)

    st.subheader("Upload unlabeled CSV for prediction")
    upload = st.file_uploader("Choose file", type=["csv"])
    if upload:
        new_df = pd.read_csv(upload)
        new_X = pd.get_dummies(new_df, drop_first=True).reindex(columns=X.columns, fill_value=0)
        preds = results[algo]["model"].predict(new_X)
        new_df[target_col+"_pred"] = preds
        st.dataframe(new_df.head())
        download_button(new_df, "predictions.csv", "Download predictions")

# ---------- 5. Clustering & Segmentation ----------
elif page == "Clustering & Segmentation":
    st.subheader("K‑means segmentation")
    feats = st.multiselect("Numeric features", ["customer_age","avg_spend_on_site","sentiment_score"], default=["customer_age","avg_spend_on_site"])
    k = st.slider("k", 2, 10, 4)
    seg, inertia = kmeans_segmentation(filtered_df.dropna(subset=feats), feats, k)
    seg_df = filtered_df.dropna(subset=feats).copy()
    seg_df["cluster"] = seg
    st.plotly_chart(px.scatter(seg_df, x=feats[0], y=feats[1], color="cluster"), use_container_width=True)
    st.write("Cluster persona summary")
    st.dataframe(seg_df.groupby("cluster")[feats].mean())
    download_button(seg_df, "clustered_data.csv", "Download labelled data")

# ---------- 6. Association & A/B ----------
elif page == "Association & A/B":
    st.subheader("Association rules")
    cols = st.multiselect("Choose categorical cols", ["segment","ancillary_purchase","booking_channel"], default=["segment","ancillary_purchase"])
    basket = filtered_df[cols].astype(str)
    # one-hot encode basket
    oh = pd.get_dummies(basket)
    freq = apriori(oh, min_support=0.05, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=0.3).sort_values("confidence", ascending=False).head(10)
    st.dataframe(rules[["antecedents","consequents","support","confidence","lift"]])

    st.subheader("Mini A/B tester")
    ab_upload = st.file_uploader("Upload two CSVs (Control & Treatment)", accept_multiple_files=True)
    if ab_upload and len(ab_upload)==2:
        dfs = [pd.read_csv(f) for f in ab_upload]
        conv_rates = [d["converted"].mean() for d in dfs]
        lift = (conv_rates[1]-conv_rates[0])/conv_rates[0]
        st.write(f"Control CR: {conv_rates[0]:.2%}, Treatment CR: {conv_rates[1]:.2%}, Lift: {lift:.2%}")
        st.bar_chart({"Control":conv_rates[0], "Treatment":conv_rates[1]})

# ---------- 7. Optimisation & Real-Time Ops ----------
elif page == "Optimisation & Real-Time Ops":
    st.subheader("Staff-hour optimiser")
    tot_staff, optim_df = build_optimisation(filtered_df)
    st.write(f"Optimal total staff hours: {tot_staff:,.0f}")
    st.plotly_chart(px.line(optim_df, x="date", y="optimal_staff", title="Optimal staff schedule"), use_container_width=True)

    st.subheader("Real‑time gauges")
    col1,col2 = st.columns(2)
    with col1:
        st.metric("Current Utilisation", f"{filtered_df['resource_utilization'].tail(1).values[0]:.0%}")
    with col2:
        st.metric("Current Delay (min)", f"{filtered_df['delay_minutes'].tail(1).values[0]:.1f}")
    st.caption("Auto‑refreshing every 15 s...")
    st_autorefresh = st.empty()
    time.sleep(0.1)

# ---------- 8. Regression & Risk/FDM ----------
else:
    st.subheader("Regression models")
    target = st.selectbox("Choose target", ["revenue","dynamic_price"])
    features = ["competitor_price","demand_forecast","units_sold","available_units"]
    X = filtered_df[features]
    y = filtered_df[target]
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    regs = {
        "Linear": __import__("sklearn.linear_model").linear_model.LinearRegression(),
        "Ridge": __import__("sklearn.linear_model").linear_model.Ridge(alpha=1.0),
        "Lasso": __import__("sklearn.linear_model").linear_model.Lasso(alpha=0.1),
        "DT": __import__("sklearn.tree").tree.DecisionTreeRegressor(max_depth=5)
    }
    res = {}
    for name, model in regs.items():
        model.fit(X_train,y_train)
        res[name] = {"R2":model.score(X_test,y_test)}
    st.dataframe(pd.DataFrame(res).T)

    st.subheader("Flight data monitoring anomalies")
    flights = filtered_df[filtered_df["property_type"]=="Flight"]
    anomalies = flights[flights["fdm_anomaly_flag"]==True]
    st.write(f"{len(anomalies)} anomalies detected")
    st.plotly_chart(px.scatter(flights, x="date", y="delay_minutes", color=flights["fdm_anomaly_flag"].map({True:'Anomaly',False:'Normal'})))