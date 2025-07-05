"""
Reusable ML / optimisation helpers for the Streamlit dashboard
"""
import pandas as pd, numpy as np, streamlit as st, plotly.express as px, plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
from prophet import Prophet
import statsmodels.api as sm
import pulp
import itertools, datetime, altair as alt

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
        results[name] = {
            "model": model,
            "train_acc": model.score(X_train, y_train),
            "test_acc": model.score(X_test, y_test),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "y_test": y_test,
            "y_pred": y_pred,
            "y_proba": getattr(model, "predict_proba", lambda X: None)(X_test)
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

@st.cache_data(show_spinner=False)
def forecast_prophet(df, target_col, periods=30):
    temp = df[["date", target_col]].rename(columns={"date":"ds", target_col:"y"})
    m = Prophet()
    m.fit(temp)
    future = m.make_future_dataframe(periods=periods)
    forecast = m.predict(future)
    return m, forecast

@st.cache_data(show_spinner=False)
def kmeans_segmentation(df, features, k):
    km = KMeans(n_clusters=k, random_state=42)
    seg = km.fit_predict(df[features])
    return seg, km.inertia_

def build_optimisation(df, demand_col="demand_forecast", staff_col="staff_hours"):
    """
    Simple LP: minimise total staff hours given demand >= units_sold forecast
    """
    model = pulp.LpProblem("StaffOptim", pulp.LpMinimize)
    staff_vars = {i: pulp.LpVariable(f"staff_{i}", lowBound=0) for i in df.index}
    # Objective
    model += pulp.lpSum([staff_vars[i] for i in df.index])
    # Demand satisfaction
    for i, row in df.iterrows():
        model += staff_vars[i] >= row[demand_col] * row["available_units"]
    model.solve(pulp.PULP_CBC_CMD(msg=0))
    df["optimal_staff"] = df.index.map(lambda i: staff_vars[i].value())
    return df["optimal_staff"].sum(), df