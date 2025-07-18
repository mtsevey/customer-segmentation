import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset (tab separated)
df = pd.read_csv('marketing_campaign.csv', sep='\t')

# Feature engineering
# Impute missing Income with median
df['Income'].fillna(df['Income'].median(), inplace=True)

# Age from Year_Birth (assuming current year 2015 since dataset around 2014)
df['Age'] = 2015 - df['Year_Birth']

# Combine kidhome and teenhome as children
df['Children'] = df['Kidhome'] + df['Teenhome']

# Convert Dt_Customer to datetime and compute tenure (days since minimum date)
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format='%d-%m-%Y', errors='coerce')
min_date = df['Dt_Customer'].min()
df['Tenure'] = (df['Dt_Customer'] - min_date).dt.days

# Drop unnecessary columns
cols_to_drop = ['ID', 'Year_Birth', 'Kidhome', 'Teenhome', 'Dt_Customer', 'Z_CostContact', 'Z_Revenue']
# Check if columns exist
cols_to_drop = [col for col in cols_to_drop if col in df.columns]
df.drop(columns=cols_to_drop, inplace=True)

# One-hot encode categorical variables
categorical_cols = ['Education', 'Marital_Status']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Scale features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df)

# Determine optimal k using silhouette for k=2 to k=6
silhouette_scores = {}
for k in range(2, 7):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled_features)
    score = silhouette_score(scaled_features, labels)
    silhouette_scores[k] = score

# Choose k with highest silhouette
best_k = max(silhouette_scores, key=silhouette_scores.get)

# Fit KMeans with best_k
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(scaled_features)

# Add cluster labels to original df
df['Cluster'] = cluster_labels

# Compute cluster summary statistics
summary = df.groupby('Cluster').agg({
    'Age': 'mean',
    'Income': 'mean',
    'Recency': 'mean',
    'MntWines': 'mean',
    'MntFruits': 'mean',
    'MntMeatProducts': 'mean',
    'MntFishProducts': 'mean',
    'MntSweetProducts': 'mean',
    'MntGoldProds': 'mean',
    'NumDealsPurchases': 'mean',
    'NumWebPurchases': 'mean',
    'NumCatalogPurchases': 'mean',
    'NumStorePurchases': 'mean',
    'NumWebVisitsMonth': 'mean',
    'Children': 'mean',
    'Tenure': 'mean'
}).reset_index()

# Save summary to CSV
summary.to_csv('cluster_summary.csv', index=False)

# Plot distribution of clusters
plt.figure(figsize=(6,4))
sns.countplot(x='Cluster', data=df)
plt.title('Customer Cluster Distribution (K={} clusters)'.format(best_k))
plt.xlabel('Cluster')
plt.ylabel('Number of Customers')
plt.tight_layout()
plt.savefig('cluster_distribution.png')
plt.close()

# PCA for visualization
pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_features)

plt.figure(figsize=(7,5))
for cluster in range(best_k):
    idx = cluster_labels == cluster
    plt.scatter(pca_components[idx, 0], pca_components[idx, 1], s=20, label=f'Cluster {cluster}')
plt.title('Customer Segments (PCA 2D)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.tight_layout()
plt.savefig('customer_segments_pca.png')
plt.close()

# Also run DBSCAN for demonstration
# Use epsilon (eps) and min_samples heuristically; scale features necessary
# Determine DBSCAN clusters
try:
    dbscan = DBSCAN(eps=0.8, min_samples=5)
    db_labels = dbscan.fit_predict(scaled_features)
    # Save DBSCAN cluster counts
    db_counts = pd.Series(db_labels).value_counts()
    db_counts.to_csv('dbscan_cluster_counts.csv')
except Exception as e:
    print(f"DBSCAN failed: {e}")

print('Analysis complete. Best k:', best_k)
