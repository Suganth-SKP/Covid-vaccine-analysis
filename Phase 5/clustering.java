import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# Load the vaccine distribution data 
data = pd.read_csv('country_vaccinations.csv')
data_copy = data.copy()

# Select relevant features for clustering
features = data[['daily_vaccinations', 'daily_vaccinations_per_million']]

# Data Preprocessing
# 1. Handling missing values (if any)
features.fillna(0, inplace=True)  # Replace missing values with zeros 

# 2. Standardization (optional but recommended)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# 3. Dimensionality Reduction (optional but recommended for high-dimensional data)
# Using Principal Component Analysis (PCA) to reduce dimensionality
pca = PCA(n_components=2)  # Adjust the number of components as needed
reduced_features = pca.fit_transform(scaled_features)

# At this point, 'reduced_features' contains the preprocessed data suitable for clustering

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(reduced_features)

# Add cluster labels to the original DataFrame
data['Cluster'] = clusters
selected_attributes = data[['daily_vaccinations', 'daily_vaccinations_per_million', 'Cluster']]
selected_attributes.to_csv('vaccine_distribution_clusters.csv', index=False)
# Print the cluster assignments
print(data[['country', 'daily_vaccinations', 'daily_vaccinations_per_million', 'Cluster']])

# Visualize the clusters
sns.scatterplot(x='daily_vaccinations', y='daily_vaccinations_per_million', hue='Cluster', data=data)
plt.xlabel('Daily Vaccinations')
plt.ylabel('Daily Vaccinations per Million')
plt.title('Clustering Results')
plt.show()
