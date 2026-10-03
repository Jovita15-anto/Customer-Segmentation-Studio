import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils import describe, get_data, numeric_cols

st.title("Predict Segment for a New Customer")
st.caption("Cluster the data with KMeans, then train a Random Forest on a train/test split "
           "so new customers can be assigned to a segment instantly.")
df, name = get_data()
num = numeric_cols(df)

features = st.sidebar.multiselect("Features", num, default=num)
k = st.sidebar.slider("Number of segments (k)", 2, 8, 4)
depth = st.sidebar.slider("Random Forest max_depth", 2, 20, 6)
if len(features) < 2:
    st.warning("Select at least two features in the sidebar.")
    st.stop()

data = df[features].dropna()
scaler = StandardScaler().fit(data)
X = scaler.transform(data)
labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)

Xtr, Xte, ytr, yte = train_test_split(X, labels, test_size=0.25, stratify=labels, random_state=42)
rf = RandomForestClassifier(n_estimators=200, max_depth=depth, random_state=42).fit(Xtr, ytr)
pred = rf.predict(Xte)

a, b = st.columns([1, 2])
a.metric("Held-out accuracy", f"{accuracy_score(yte, pred):.1%}")
a.caption("Labels come from KMeans on the same features, so high accuracy is expected – "
          "this model is a fast 'segment scorer', not a forecast.")
b.write("**Feature importance**")
b.bar_chart(pd.Series(rf.feature_importances_, index=features).sort_values())
st.dataframe(pd.DataFrame(classification_report(yte, pred, output_dict=True)).T.round(2), width="stretch")

st.subheader("Score a new customer")
profiles = describe(data, labels)
with st.form("new_customer"):
    cols = st.columns(3)
    vals = {f: cols[i % 3].number_input(f, value=float(data[f].median())) for i, f in enumerate(features)}
    go = st.form_submit_button("Predict segment")
if go:
    row = scaler.transform(pd.DataFrame([vals])[features])
    seg = int(rf.predict(row)[0])
    proba = rf.predict_proba(row)[0]
    st.success(f"Predicted segment: **Cluster {seg}** ({proba.max():.0%} confidence) – {profiles[seg]}")
