import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.neighbors import NearestNeighbors

from utils import describe, fit_labels, get_data, k_sweep, numeric_cols, pca_2d, prepare, scatter, score

st.title("Cluster Explorer")
df, name = get_data()
num = numeric_cols(df)
st.caption(f"Dataset: {name} · {len(df)} rows")

st.sidebar.header("Settings")
features = st.sidebar.multiselect("Features", num, default=num)
scale = st.sidebar.checkbox("Standardize features", True, help="Strongly recommended: stops large-valued columns dominating.")
algo = st.sidebar.selectbox("Algorithm", ["KMeans", "DBSCAN", "Hierarchical", "Gaussian Mixture"])
k, eps, ms, linkage = 4, 0.5, 5, "ward"
if algo == "DBSCAN":
    eps = st.sidebar.number_input("eps", 0.05, 10.0, 0.8, 0.05)
    ms = st.sidebar.number_input("min_samples", 2, 100, 5, 1)
else:
    k = st.sidebar.slider("Number of clusters (k)", 2, 10, 4)
    if algo == "Hierarchical":
        linkage = st.sidebar.selectbox("Linkage", ["ward", "complete", "average", "single"])

if len(features) < 2:
    st.warning("Select at least two features in the sidebar.")
    st.stop()

data, X = prepare(df, features, scale)
labels = fit_labels(algo, X, k, eps, ms, linkage)
m = score(X, labels)

c = st.columns(4)
c[0].metric("Clusters", m["clusters"])
c[1].metric("Silhouette ↑", m["silhouette"])
c[2].metric("Davies-Bouldin ↓", m["davies_bouldin"])
c[3].metric("Noise points", f"{m['noise %']}%")

t1, t2, t3, t4 = st.tabs(["Clusters", "Tune parameters", "Segment profiles", "Download"])

with t1:
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter(ax, pca_2d(X), labels, f"{algo} (PCA projection)")
    ax.legend()
    st.pyplot(fig)

with t2:
    if algo == "DBSCAN":
        st.write("**k-distance plot** – pick `eps` near the 'knee' of this curve (k = min_samples).")
        d, _ = NearestNeighbors(n_neighbors=int(ms)).fit(X).kneighbors(X)
        fig, ax = plt.subplots()
        ax.plot(np.sort(d[:, -1]))
        ax.set_xlabel("Points (sorted)")
        ax.set_ylabel(f"Distance to {int(ms)}th neighbour")
        st.pyplot(fig)
    else:
        sweep = k_sweep(X)
        a, b = st.columns(2)
        a.write("**Elbow method** (KMeans inertia)")
        a.line_chart(sweep.set_index("k")["inertia"])
        b.write("**Silhouette score** (higher is better)")
        b.line_chart(sweep.set_index("k")["silhouette"])
        st.info(f"Best silhouette at k = {int(sweep.loc[sweep.silhouette.idxmax(), 'k'])}")

with t3:
    prof = data.assign(Cluster=labels).groupby("Cluster").mean().round(2)
    prof.insert(0, "Size", pd.Series(labels).value_counts().sort_index())
    st.dataframe(prof, width="stretch")
    for lab, text in describe(data, labels).items():
        st.markdown(f"**{'Noise' if lab == -1 else f'Cluster {lab}'}** ({(labels == lab).sum()} customers) – {text}")
    z = (data - data.mean()) / data.std()
    zc = z.assign(Cluster=labels).groupby("Cluster").mean()
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(zc) + 2))
    sns.heatmap(zc, annot=True, fmt=".1f", cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Cluster mean vs overall (z-score)")
    st.pyplot(fig)

with t4:
    out = df.loc[data.index].assign(cluster=labels)
    st.dataframe(out.head(), width="stretch")
    st.download_button("Download labelled data (CSV)", out.to_csv(index=False), "segmented_customers.csv", "text/csv")
