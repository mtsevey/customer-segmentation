# Customer Segmentation Project

This project analyzes a **marketing campaign dataset** containing **2,240 customers** and 31 variables covering demographic and behavioral information【357353303844640†L31-L35】.  The dataset includes customer demographics (year of birth, education, marital status and income), household composition (number of kids and teens), the date when each customer joined, recency (days since the last purchase) and spending amounts across various product categories (wines, meat, fish, sweets and gold products).  It also records the number of deals and purchases through web, catalogue and store channels, the number of website visits per month and whether each customer accepted previous campaigns【357353303844640†L31-L35】.  These rich attributes make the data ideal for **unsupervised customer segmentation**.

## Repository contents

| File | Description |
|---|---|
| **marketing_campaign.csv** | Raw dataset (tab‑separated). Contains customer demographics, purchases and campaign response information. |
| **customer_segmentation_analysis.py** | Python script that performs data cleaning, feature engineering, K‑means clustering and basic DBSCAN clustering.  It outputs summary tables and visualizations. |
| **Customer_Segmentation.ipynb** | Jupyter notebook that documents the analysis and visualizes the results step by step. |
| **cluster_summary.csv** | Summary statistics (mean age, income, spending, etc.) for each K‑means segment. |
| **cluster_distribution.png** | Bar chart showing how many customers fall into each segment. |
| **customer_segments_pca.png** | PCA scatter plot colored by cluster labels. |

## Methodology

1. **Data preparation** – The script reads the TSV dataset and imputes missing income values with the median.  New features are engineered: **Age** (using an approximate reference year 2015), **Children** (sum of kids and teens at home) and **Tenure** (days since the earliest acquisition date).  Columns such as `ID`, `Year_Birth`, `Kidhome`, `Teenhome` and the original date variables are dropped.  Categorical variables (`Education` and `Marital_Status`) are one‑hot encoded, and numeric features are standardized.

2. **Choosing the number of clusters** – K‑means clustering is evaluated for 2–6 clusters using the **silhouette score** to measure how well each data point fits within its cluster relative to other clusters.  The highest silhouette score occurred at **k = 2**, indicating two distinct customer segments.

3. **Clustering models** – A K‑means model with k = 2 was fitted to the standardized features.  A DBSCAN model was also tested for comparison but labelled all points as noise under default parameters, so the focus of the analysis remains on the K‑means results.

4. **Evaluation and visualization** – The cluster assignment is added to the dataset and a summary table is computed.  A **cluster distribution chart** shows the number of customers in each cluster, and a **PCA scatter plot** visualizes the segmentation in two dimensions.

## Findings

Two distinct customer segments emerged:

* **Cluster 0 – high‑value customers**:  Customers in this group are on average older (≈ 48 years) and earn a higher income (≈ \$72 k).  They spend far more on wines, meat and gold products, make many catalogue and store purchases, and usually have no children.  They also have shorter tenures.  **Marketing strategy:** focus on premium product offers, personalized loyalty programs and exclusive experiences to retain these high‑value clients.

* **Cluster 1 – budget‑conscious families**:  This cluster consists of slightly younger customers (≈ 45 years) with lower incomes (≈ \$39 k) who spend less across all categories.  They have more children, visit the web frequently and take advantage of deals.  **Marketing strategy:** offer targeted discounts, family‑friendly bundles and convenience‑oriented online promotions to increase engagement and spend.

While DBSCAN did not identify additional structure, the K‑means segments provide actionable insights that can drive **personalized marketing strategies**.

## Usage

You can reproduce the analysis by running the Python script:

```bash
python customer_segmentation_analysis.py
```

This will generate the cluster summary and images.  Alternatively, open the Jupyter notebook **Customer_Segmentation.ipynb** to step through the analysis interactively.

## References

* Gigasheet – Example Marketing Campaign Data: the dataset contains 31 columns and 2,240 rows of customer demographics, purchasing behavior and campaign response data【357353303844640†L31-L35】.

