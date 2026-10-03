# 🧩 Customer Segmentation Studio

An interactive **machine learning web application** built with Streamlit that helps users analyze customer data, discover meaningful customer segments, compare clustering algorithms, understand segment characteristics, and assign new customers to existing segments.

The application supports multiple unsupervised learning techniques and provides visual and statistical insights so that users can make data-driven decisions from customer data.

---

## 📌 Project Overview

Customer segmentation is the process of grouping customers with similar characteristics or behaviors.

This project allows a user to:

* Upload their own customer dataset
* Select numerical features for analysis
* Standardize features before clustering
* Apply multiple clustering algorithms
* Visualize customer groups using PCA
* Evaluate clustering quality using multiple metrics
* Understand the characteristics of each segment
* Download segmented customer data
* Predict the segment of a new customer

The project is designed as an interactive **Data Science + Machine Learning application** rather than a simple standalone clustering script.

---

## 🎯 Problem Statement

Businesses often have large amounts of customer data but may not know how to identify groups of customers with similar purchasing behavior.

Manually analyzing every customer can be difficult and time-consuming.

This application provides an interactive way to discover customer groups automatically using machine learning and convert the resulting clusters into understandable customer profiles.

---

## 💡 What Problem Does It Solve?

The application can help users answer questions such as:

* Which customers have similar behavior?
* How many meaningful customer segments exist?
* Which clustering algorithm produces useful groups?
* Which features distinguish one segment from another?
* Which customers may be high-value or at-risk?
* What segment does a new customer belong to?

---

## ✨ Key Features

### 1. 📂 Customer Data Upload

Users can upload:

* CSV files
* Excel (`.xlsx`) files

A built-in synthetic customer dataset is also provided for testing the application without uploading a file.

---

### 2. 🧩 Cluster Explorer

The Cluster Explorer allows users to experiment with different clustering techniques.

Supported algorithms:

* **KMeans**
* **DBSCAN**
* **Hierarchical Clustering**
* **Gaussian Mixture Model**

Users can select:

* Features
* Number of clusters
* DBSCAN `eps`
* DBSCAN `min_samples`
* Hierarchical linkage method
* Feature standardization

---

### 3. 📊 Cluster Visualization

The application uses **Principal Component Analysis (PCA)** to reduce high-dimensional customer data to two dimensions.

This makes it possible to visualize customer clusters in a 2D scatter plot.

**Flow:**

```text
Customer Features
       ↓
Feature Scaling
       ↓
PCA
       ↓
2D Representation
       ↓
Cluster Visualization
```

---

### 4. 📈 Clustering Evaluation

The application evaluates clustering quality using:

| Metric                  | Meaning                                  | Preferred Direction |
| ----------------------- | ---------------------------------------- | ------------------- |
| Silhouette Score        | Measures how well-separated clusters are | Higher              |
| Davies-Bouldin Index    | Measures cluster similarity              | Lower               |
| Calinski-Harabasz Score | Measures cluster separation              | Higher              |

For DBSCAN, noise points are excluded from the clustering validity calculations.

---

### 5. 🔧 Parameter Tuning

The application provides different tools depending on the selected algorithm.

For KMeans, Hierarchical Clustering and Gaussian Mixture:

* Elbow method
* Silhouette analysis
* Automatic identification of the best silhouette value

For DBSCAN:

* k-distance plot
* `eps` tuning
* `min_samples` tuning

---

### 6. 👥 Segment Profiles

After clustering, the application generates a summary for every customer segment.

It displays:

* Number of customers
* Average feature values
* Features significantly above the overall average
* Features significantly below the overall average
* Cluster-level z-score heatmap

Example:

```text
Cluster 0
High annual_spend · High orders_per_year

Cluster 1
Low annual_spend · High days_since_last_purchase
```

This converts numerical clustering results into more understandable segment descriptions.

---

### 7. 📥 Export Segmented Data

Users can download the resulting dataset with an additional `cluster` column.

Example:

```text
customer_id | annual_spend | orders_per_year | cluster
-------------------------------------------------------
1001        | 12500        | 18              | 2
1002        | 3200         | 5               | 0
1003        | 8900         | 12              | 1
```

The output can be downloaded as a CSV file.

---

### 8. 🔬 Compare Clustering Algorithms

The Compare Algorithms page evaluates all four algorithms on the same dataset.

Algorithms compared:

```text
KMeans
DBSCAN
Hierarchical Clustering
Gaussian Mixture
```

The application displays:

* Number of clusters
* Noise percentage
* Silhouette score
* Davies-Bouldin score
* Calinski-Harabasz score
* PCA-based cluster visualizations

This allows users to understand how different clustering approaches behave on the same customer data.

---

### 9. 🔮 Predict Segment for a New Customer

The application also provides a way to assign a new customer to an existing segment.

The workflow is:

```text
Customer Dataset
       ↓
Standardization
       ↓
KMeans Clustering
       ↓
Generate Cluster Labels
       ↓
Train Random Forest
       ↓
New Customer Data
       ↓
Predicted Segment
```

A Random Forest classifier is trained using the KMeans-generated labels.

The page also displays:

* Held-out accuracy
* Classification report
* Feature importance
* Predicted cluster
* Prediction confidence
* Segment description

> **Note:** The Random Forest is acting as a fast segment scorer. It is not predicting future customer behavior; it is learning to reproduce the segmentation created by KMeans.

---

## 🏗️ High-Level Flow

```text
User
  ↓
Streamlit Web Interface
  ↓
Upload Customer Dataset
  ↓
Data Validation & Feature Selection
  ↓
Feature Standardization
  ↓
┌─────────────────────────────────────┐
│       Clustering Algorithms         │
│                                     │
│ KMeans                              │
│ DBSCAN                              │
│ Hierarchical Clustering             │
│ Gaussian Mixture                    │
└─────────────────────────────────────┘
  ↓
Cluster Evaluation
  ↓
PCA Visualization
  ↓
Segment Profiles
  ↓
Download Segmented Data
  ↓
New Customer → Segment Prediction
```

---

## 🛠️ Tech Stack

### Programming Language

* Python

### User Interface

* Streamlit

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-learn

### Clustering

* KMeans
* DBSCAN
* Agglomerative / Hierarchical Clustering
* Gaussian Mixture Model

### Dimensionality Reduction

* PCA

### Visualization

* Matplotlib
* Seaborn
* Streamlit charts

### Data Input

* CSV
* Excel

---

## 📁 Project Structure

```text
cluster/
│
├── app.py
├── utils.py
├── generate_data.py
├── requirements.txt
├── README.md
│
├── data/
│   └── customers.csv
│
└── pages/
    ├── 1_Cluster_Explorer.py
    ├── 2_Compare_Algorithms.py
    └── 3_Predict_Segment.py
```

### File Description

| File                      | Purpose                                                        |
| ------------------------- | -------------------------------------------------------------- |
| `app.py`                  | Main Streamlit application and dataset upload                  |
| `utils.py`                | Data processing, clustering, evaluation and plotting functions |
| `generate_data.py`        | Generates the synthetic customer dataset                       |
| `1_Cluster_Explorer.py`   | Interactive clustering and segment analysis                    |
| `2_Compare_Algorithms.py` | Compares the four clustering algorithms                        |
| `3_Predict_Segment.py`    | Predicts the segment of a new customer                         |
| `customers.csv`           | Sample customer dataset                                        |
| `requirements.txt`        | Python dependencies                                            |

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <(https://github.com/Jovita15-anto/Customer-Segmentation-Studio.git)>
cd cluster
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

#### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

#### Windows Command Prompt

```cmd
.venv\Scripts\activate
```

---

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Streamlit application with:

```bash
python -m streamlit run app.py
```

Using `python -m streamlit` is useful on Windows when the `streamlit` command itself is not recognized.

The application will open in your browser.

---

## 📊 Sample Dataset

The project includes a synthetic customer dataset containing RFM-style customer behavior features.

Example features include:

* `days_since_last_purchase`
* `orders_per_year`
* `annual_spend`

The dataset is provided for demonstration and testing purposes.

You can also generate a fresh sample dataset using:

```bash
python generate_data.py
```

---

## 📸 Screenshots

Add screenshots of the main application pages here.

### Home / Dataset Upload

```text
screenshots/home.png
```

### Cluster Explorer

```text
screenshots/cluster-explorer.png
```

### Algorithm Comparison

```text
screenshots/compare-algorithms.png
```

### Segment Prediction

```text
screenshots/predict-segment.png
```

---

## 📌 Example Use Case

A business uploads customer transaction data containing:

```text
Customer
Purchase Frequency
Annual Spend
Recency
```

The application analyzes the data and discovers customer groups.

For example, the resulting segments may represent customers with:

* High spending and frequent purchases
* Moderate spending and regular purchases
* Low spending and infrequent purchases
* Customers who have not purchased recently

The exact segments depend on the uploaded dataset and selected clustering configuration.

---

## 🧠 Machine Learning Concepts Used

This project demonstrates practical implementation of several Data Science concepts:

### Unsupervised Learning

Customer segments are discovered without predefined labels.

### Feature Scaling

Standardization prevents features with larger numerical ranges from dominating distance-based algorithms.

### Clustering

Different algorithms identify groups using different mathematical approaches.

### Dimensionality Reduction

PCA reduces multiple features into two dimensions for visualization.

### Model Evaluation

Internal clustering metrics are used to assess cluster structure.

### Supervised Learning

Random Forest is used as a segment scorer for assigning new customers to previously discovered KMeans segments.

---

## 📈 Sample Findings

On the bundled synthetic dataset, clustering behavior varies depending on the algorithm and parameter settings.

For example:

* KMeans can provide relatively compact customer groups when the number of clusters is chosen appropriately.
* DBSCAN can identify dense groups and treat unusual observations as noise.
* Hierarchical clustering provides an alternative grouping strategy based on cluster linkage.
* Gaussian Mixture provides probabilistic cluster assignments.

Feature standardization is particularly important because customer features can have very different numerical scales.

---

## ⚠️ Limitations

* The bundled dataset is synthetic.
* Clustering quality depends heavily on feature selection.
* Different algorithms may produce different segment structures.
* DBSCAN is sensitive to `eps` and `min_samples`.
* PCA visualizations are only a 2D representation of potentially higher-dimensional data.
* The Random Forest segment predictor learns from KMeans-generated labels rather than independently validated business labels.

---

## 🚀 Future Enhancements

Possible future improvements include:

1. **Automated optimal cluster selection** using multiple validation metrics.
2. **Interactive customer profiling** with detailed segment-level insights.
3. **Real-world RFM analysis** using transaction-level datasets.
4. **Business-oriented segment naming** such as High-Value, Loyal, At-Risk, and New Customers.
5. **Interactive Plotly visualizations** for better exploration of customer segments.
6. **Historical segmentation tracking** to monitor how customers move between segments.
7. **Deployment with user authentication and persistent datasets** for multi-user usage.

---

## 🎓 Learning Outcomes

Through this project, the following concepts were implemented:

* Data preprocessing
* Exploratory data analysis
* Feature scaling
* Unsupervised machine learning
* Multiple clustering algorithms
* PCA
* Cluster validation metrics
* Hyperparameter tuning
* Data visualization
* Model interpretation
* Supervised learning
* Streamlit application development
* CSV/Excel data handling

---

## 👩‍💻 Author

**Anto Jovita J**

B.Tech — Artificial Intelligence & Data Science

Interested in:

* Data Science
* Machine Learning
* Artificial Intelligence
* Frontend Development
* UI/UX Design


