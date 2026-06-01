# High-Value E-Commerce Customer Segmentation & Behavioral Analytics
An End-to-End Data Science Pipeline leveraging RFM Metrics and Unsupervised Machine Learning (K-Means Clustering) to optimize customer retention and maximize Lifetime Value (LTV).

---

## 📌 Project Architecture & Objectives
In corporate retail environments, marketing teams struggle to extract actionable targeting strategies from millions of raw transaction lines. This project establishes an automated analytics pipeline that transforms messy, transaction-level invoice data into highly targeted customer personas.

### Core Objectives:
* Implement robust preprocessing to clean and structurally fix international transactional logs.
* Engineer an **RFM (Recency, Frequency, Monetary)** behavioral framework.
* Apply Advanced Statistical Scaling and **Unsupervised Machine Learning** to classify users without human bias.
* Synthesize algorithmic findings into data-driven corporate recommendations.

---

## 🛠️ Enterprise Tech Stack
* **Core Language:** Python 3.x
* **Data Engineering & Wrangling:** Pandas, NumPy
* **Statistical Modeling & ML:** Scikit-Learn (`StandardScaler`, `KMeans`)
* **Data Visualization Suites:** Seaborn, Matplotlib
* **Development Environment:** Jupyter Notebook (Anaconda Data Science Ecosystem)

---

## ⚙️ Data Engineering & Pipeline Execution

### Phase 1: Ingestion & Quality Control (Data Cleaning)
* **Dataset Scale:** 541,909 raw transaction entries from an international UK-based digital retailer.
* **Character Encoding Resolution:** Configured data ingestion via `ISO-8859-1` to handle European character anomalies without crashing runtime execution.
* **Missing Value Rectification:** Purged **135,080 records** lacking a structural `CustomerID`—ensuring all downstream analytical lines map uniquely to an authentic consumer account.
* **Anomalous Outlier Filtration:** Analyzed data patterns to detect and filter out **8,905 cancelled orders** (invoices prefixed with 'C' carrying negative values) and promotional freebies (`UnitPrice = 0`).
* **Feature Engineering:** Derived a deterministic line-item financial metric:
  $$\text{TotalRevenue} = \text{Quantity} \times \text{UnitPrice}$$
* **Clean Baseline Data Shape:** Reduced and optimized the dataset to **397,884 pristine transactional rows**.

### Phase 2: Behavioral Feature Engineering (RFM Model)
The linear transaction records were aggregated into individual human profiles ($N = 4,338$ unique customers) by establishing a temporal snapshot baseline one day past the latest recorded purchase (`2011-12-10`). Three custom behavior features were engineered per user:
1. **Recency ($R$):** Count of days elapsed between the customer's maximum `InvoiceDate` and the snapshot baseline.
2. **Frequency ($F$):** Total count of unique, distinct `InvoiceNo` entries submitted by the customer.
3. **Monetary ($M$):** Total aggregated financial capital spent ($\sum \text{TotalRevenue}$) per customer.

### Phase 3: Mathematical Normalization & Clustering
* **The Log-Transformation:** Behavioral distribution data exhibits a massive right-skew (long tails from high-spending outliers). To prevent large scale distortions in distance metrics, a logarithmic transformation was executed:
  $$X_{\text{log}} = \ln(X + 1)$$
* **Z-Score Feature Scaling:** Because K-Means relies on Euclidean geometric distances, variations in units (Days vs. Pounds) skew clusters. Features were scaled using the `StandardScaler` to force a mean of $0$ and a standard deviation of $1$:
  $$Z = \frac{x - \mu}{\sigma}$$
* **Algorithmic Clustering:** Initialized and fit a **K-Means Clustering** model ($k=3$) to partition data space into minimum variance hyper-spheres.

---

## 📊 Strategic Findings & Macro Insights

### Macro Business Discoveries:
* **Geographic Consolidation:** The retail infrastructure is highly centralized. The **United Kingdom** acts as the primary revenue generator contributing **£7,308,391.55**. High-performing secondary scaling markets were isolated in Western Europe: Netherlands (£285.4k), EIRE (£265.5k), Germany (£228.8k), and France (£209.0k).
* **Inventory Demand Drivers:** "PAPER CRAFT, LITTLE BIRDIE" and "MEDIUM CERAMIC TOP STORAGE JAR" were isolated as the highest volume inventory assets transacted across the dataset lifecycle.

### Algorithmic Segment Profiling (K-Means Results):

| Cluster Identity | Recency (Mean) | Frequency (Mean Orders) | Monetary Spend (Mean) | Core Corporate Strategy |
| :--- | :--- | :--- | :--- | :--- |
| **Cluster 0: High-Value VIPs** | 17.0 Days | 13.3 Orders | **£7,908.95** | **Protect & Reward:** Enroll in immediate high-tier loyalty loops, offer early product rollouts, and assign premier customer care. |
| **Cluster 2: Steady Core Retail** | 44.2 Days | 3.4 Orders | **£1,267.38** | **Cross-Sell / Upsell:** Leverage personalized algorithmic product recommendations to accelerate order frequency. |
| **Cluster 1: Dormant / High-Risk** | 167.4 Days | 1.3 Orders | **£362.54** | **Win-Back Campaigns:** Allocate high-incentive promotional discounts to address churn risk before complete account attrition. |

---

## 📂 Repository Layout
* `Online_Retail_Customer_Segmentation.ipynb` -> Comprehensive, fully documented Jupyter Notebook containing data pipeline engineering and machine learning structures.
* `online_retail.csv` -> **(Local File - Excluded from Git)** The original source dataset downloaded from Kaggle/UCI. Keep this file in your root working directory to execute the notebook.
* `README.md` -> High-impact executive portfolio documentation.
