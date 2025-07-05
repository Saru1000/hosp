import pandas as pd, numpy as np, streamlit as st, plotly.express as px, plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pulp

# ---------------- Classification ----------------
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
        proba = model.predict_proba(X_test) if hasattr(model,"predict_proba") else None
        results[name] = dict(
            model=model,
            train_acc=accuracy_score(y_train, model.predict(X_train)),
            test_acc=accuracy_score(y_test, y_pred),
            precision=precision_score(y_test, y_pred, zero_division=0),
            recall=recall_score(y_test, y_pred, zero_division=0),
            f1=f1_score(y_test, y_pred, zero_division=0),
            y_test=y_test, y_pred=y_pred, y_proba=proba
        )
    return results

def plot_confusion(y_true, y_pred, labels):
    fig = px.imshow(confusion_matrix(y_true,y_pred,labels=labels),
                    x=labels, y=labels, text_auto=True,
                    color_continuous_scale='Blues', title="Confusion Matrix")
    return fig

def plot_roc(results):
    fig = go.Figure()
    for name,res in results.items():
        if res["y_proba"] is None: continue
        fpr, tpr, _ = roc_curve(res["y_test"], res["y_proba"][:,1])
        fig.add_trace(go.Scatter(x=fpr,y=tpr,mode='lines',name=name))
    fig.update_layout(title="ROC Curves", xaxis_title="FPR", yaxis_title="TPR")
    return fig

# ---------------- Forecasting ----------------
@st.cache_data(show_spinner=False)
def forecast_arima(ts, steps=30):
    model = ARIMA(ts, order=(2,1,2)).fit()
    fc = model.get_forecast(steps=steps)
    idx = pd.date_range(ts.index.max()+pd.Timedelta(days=1), periods=steps, freq='D')
    df_fc = pd.DataFrame({
        "ds": idx,
        "yhat": fc.predicted_mean,
        "yhat_lower": fc.conf_int().iloc[:,0],
        "yhat_upper": fc.conf_int().iloc[:,1]
    })
    return df_fc

@st.cache_data(show_spinner=False)
def forecast_hwes(ts, steps=30):
    model = ExponentialSmoothing(ts, trend='add', seasonal=None).fit()
    idx = pd.date_range(ts.index.max()+pd.Timedelta(days=1), periods=steps, freq='D')
    preds = model.forecast(steps)
    df_fc = pd.DataFrame({
        "ds": idx,
        "yhat": preds,
        "yhat_lower": preds*0.95,
        "yhat_upper": preds*1.05
    })
    return df_fc

# ---------------- Clustering ----------------
@st.cache_data(show_spinner=False)
def kmeans_segmentation(df_num, k):
    km = KMeans(n_clusters=k, random_state=42)
    seg = km.fit_predict(df_num)
    return seg, km.inertia_

# ---------------- Optimisation ----------------
def optimise_staff(df, demand_col="demand_forecast"):
    prob = pulp.LpProblem("StaffMin", pulp.LpMinimize)
    staff_vars = {i:pulp.LpVariable(f"s{i}", lowBound=0) for i in df.index}
    prob += pulp.lpSum(staff_vars.values())
    for i,row in df.iterrows():
        prob += staff_vars[i] >= row[demand_col] * row["available_units"]
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    df["optimal_staff"] = [staff_vars[i].value() for i in df.index]
    return df["optimal_staff"].sum(), df