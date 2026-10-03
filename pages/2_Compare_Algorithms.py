import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from utils import ALGOS, fit_labels, get_data, numeric_cols, pca_2d, prepare, scatter, score

st.title("Compare Algorithms")
df, name = get_data()
num = numeric_cols(df)

st.sidebar.header("Settings")
features = st.sidebar.multiselect("Features", num, default=num)
k = st.sidebar.slider("k (KMeans / Hierarchical / GMM)", 2, 10, 4)
eps = st.sidebar.number_input("DBSCAN eps", 0.05, 10.0, 0.8, 0.05)
ms = st.sidebar.number_input("DBSCAN min_samples", 2, 100, 5, 1)

if len(features) < 2:
    st.warning("Select at least two features in the sidebar.")
    st.stop()

data, X = prepare(df, features, True)
coords = pca_2d(X)

rows = []
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, algo in zip(axes.ravel(), ALGOS):
    labels = fit_labels(algo, X, k=k, eps=eps, min_samples=ms)
    rows.append({"Algorithm": algo, **score(X, labels)})
    scatter(ax, coords, labels, algo)
fig.tight_layout()

res = pd.DataFrame(rows).set_index("Algorithm")
st.dataframe(res, width="stretch")
if res["silhouette"].notna().any():
    st.success(f"Highest silhouette: **{res['silhouette'].idxmax()}**. "
               "Check the noise % too – DBSCAN scores exclude the points it discards.")
st.pyplot(fig)
st.caption("Silhouette ↑ better · Davies-Bouldin ↓ better · Calinski-Harabasz ↑ better.")
