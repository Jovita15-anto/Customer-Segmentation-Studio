import pandas as pd
import streamlit as st

from utils import get_data, numeric_cols, set_data

st.set_page_config(page_title="Customer Segmentation Studio", page_icon="🧩", layout="wide")

st.title("🧩 Customer Segmentation Studio")
st.write(
    "Upload customer data (or use the built-in sample), group customers with several "
    "clustering algorithms, compare them with validity metrics, and turn the result into "
    "plain-English segment profiles."
)

file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])
if file is not None:
    new = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
    set_data(new, file.name)

df, name = get_data()
c1, c2 = st.columns([4, 1])
c1.success(f"Active dataset: **{name}** ({df.shape[0]} rows × {df.shape[1]} columns)")
if c2.button("Reset to sample data"):
    st.session_state.pop("df", None)
    st.rerun()

st.dataframe(df.head(20), width="stretch")
st.caption(f"Numeric columns available for clustering: {', '.join(numeric_cols(df)) or 'none'}")

st.markdown(
    "**Pages (sidebar):** 1️⃣ Cluster Explorer · 2️⃣ Compare Algorithms · 3️⃣ Predict Segment"
)
