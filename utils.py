from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

SAMPLE = Path(__file__).parent / "data" / "customers.csv"
ALGOS = ["KMeans", "DBSCAN", "Hierarchical", "Gaussian Mixture"]


# ---------- data handling ----------
def set_data(df, name):
    st.session_state["df"] = df
    st.session_state["name"] = name


def get_data():
    """Dataset shared by all pages (uploaded on Home, falls back to the sample)."""
    if "df" not in st.session_state:
        set_data(pd.read_csv(SAMPLE), "sample_customers.csv")
    return st.session_state["df"], st.session_state["name"]


def numeric_cols(df):
    return df.select_dtypes("number").columns.tolist()


def prepare(df, features, scale=True):
    data = df[features].dropna()
    X = StandardScaler().fit_transform(data) if scale else data.values.astype(float)
    return data, X


# ---------- modelling ----------
def fit_labels(algo, X, k=4, eps=0.5, min_samples=5, linkage="ward"):
    if algo == "KMeans":
        return KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
    if algo == "DBSCAN":
        return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    if algo == "Hierarchical":
        return AgglomerativeClustering(n_clusters=k, linkage=linkage).fit_predict(X)
    return GaussianMixture(n_components=k, random_state=42).fit_predict(X)


def score(X, labels):
    """Internal validity metrics. DBSCAN noise (-1) is excluded from scoring."""
    mask = labels != -1
    n = len(set(labels[mask]))
    out = {"clusters": n, "noise %": round(100 * (~mask).mean(), 1),
           "silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan}
    if n >= 2 and mask.sum() > n:
        out["silhouette"] = round(silhouette_score(X[mask], labels[mask]), 3)
        out["davies_bouldin"] = round(davies_bouldin_score(X[mask], labels[mask]), 3)
        out["calinski_harabasz"] = round(calinski_harabasz_score(X[mask], labels[mask]), 1)
    return out


@st.cache_data
def k_sweep(X, k_max=10):
    rows = []
    for k in range(2, k_max + 1):
        m = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        rows.append((k, m.inertia_, silhouette_score(X, m.labels_)))
    return pd.DataFrame(rows, columns=["k", "inertia", "silhouette"])


def pca_2d(X):
    return X[:, :2] if X.shape[1] == 2 else PCA(n_components=2, random_state=42).fit_transform(X)


def describe(data, labels, top=2, thr=0.5):
    """Plain-English profile per cluster: which features are well above / below average."""
    z = (data - data.mean()) / data.std()
    out = {}
    for lab in sorted(set(labels)):
        zc = z[labels == lab].mean().sort_values()
        high = [f"{c}" for c in zc[zc > thr].index[::-1][:top]]
        low = [f"{c}" for c in zc[zc < -thr].index[:top]]
        parts = ([f"**high** {', '.join(high)}"] if high else []) + ([f"**low** {', '.join(low)}"] if low else [])
        out[lab] = " · ".join(parts) if parts else "close to average on every feature"
    return out


# ---------- plotting ----------
def scatter(ax, coords, labels, title=""):
    for lab in sorted(set(labels)):
        m = labels == lab
        color = "lightgray" if lab == -1 else plt.cm.tab10(lab % 10)
        ax.scatter(coords[m, 0], coords[m, 1], s=18, alpha=0.8, color=color,
                   label="Noise" if lab == -1 else f"Cluster {lab}")
    ax.set_title(title)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
