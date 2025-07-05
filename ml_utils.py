"""
Reusable ML / optimisation helpers for the Streamlit dashboard
"""
import pandas as pd, numpy as np, streamlit as st, plotly.express as px, plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_curve, confusion_matrix)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import pulp

# ------------------ Classification ------------------
@st.cache_data(show_spinner=False)
def train_classifiers(X, y, random_state=42):
    models = {
        "KNN": KNeighborsClassifier(),
        "DecisionTree": DecisionTreeClassifier(random_state=random_state),
        "RandomForest": RandomForestClassifier(random_state=random_state),
        "GradientBoosting": GradientBoostingClassifier(random_state=random_state)
    }
    results = {}
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y)
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        results[name] = {
            "model": model,
            "train_acc": accuracy_score(y_train, model.predict(X_train)),
            "test_acc": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "y_test": y_test, "y_pred": y_pred, "y_proba": prob
        }
    return results

def plot_confusion(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                    x=labels, y=labels, title="Confusion Matrix")
    return fig

def plot_roc(results):
    fig = go.Figure()
    for name, res in results.items():
        if res["y_proba"] is None: continue
        fpr, tpr, _ = roc_curve(res["y_test"], res["y_proba"][:,1])
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=name))
    fig.update_layout(title="ROC Curves", xaxis_title="FPR", yaxis_title="TPR")
    return fig

# ------------------ Forecasting ------------------
@st.cache_data(show_spinner=False)
def forecast_prophet(df, target_col, periods=30):
    data = df[["date", target_col]].rename(columns={"date":"ds", target_col:"y"})
    m = Prophet()
    m.fit(data)
    future = m.make_future_dataframe(periods=periods)
    forecast = m.predict(future)
    return forecast[["ds","yhat","yhat_lower","yhat_upper"]]

@st.cache_data(show_spinner=False)
def forecast_arima(df, target_col, periods=30):
    ts = df.set_index("date")[target_col].asfreq("D").interpolate()
    model = ARIMA(ts, order=(2,1,2))
    fit = model.fit()
    preds = fit.get_forecast(steps=periods)
    fc_index = pd.date_range(ts.index.max()+pd.Timedelta(days=1), periods=periods, freq="D")
    fc_series = preds.predicted_mean
    ci = preds.conf_int()
    out = pd.DataFrame({
        "ds": fc_index,
        "yhat": fc_series,
        "yhat_lower": ci.iloc[:,0],
        "yhat_upper": ci.iloc[:,1]
    })
    return out

# ------------------ Clustering ------------------
@st.cache_data(show_spinner=False)
def kmeans_segmentation(df, features, k):
    km = KMeans(n_clusters=k, random_state=42)
    seg = km.fit_predict(df[features])
    return seg, km.inertia_

# ------------------ Optimisation ------------------
def build_optimisation(df, demand_col="demand_forecast"):
    model = pulp.LpProblem("StaffOptim", pulp.LpMinimize)
    staff_vars = {i: pulp.LpVariable(f"staff_{i}", lowBound=0) for i in df.index}
    model += pulp.lpSum(staff_vars.values())
    for i,row in df.iterrows():
        model += staff_vars[i] >= row[demand_col] * row["available_units"]
    model.solve(pulp.PULP_CBC_CMD(msg=0))
    df["optimal_staff"] = [staff_vars[i].value() for i in df.index]
    return df["optimal_staff"].sum(), df