import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = {
    'CustomerID': range(1, 11),
    'Product1': np.random.randint(0, 10, 10),
    'Product2': np.random.randint(0, 10, 10),
    'Product3': np.random.randint(0, 10, 10)
}
df = pd.DataFrame(data)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df.drop('CustomerID', axis=1))
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(scaled_data)
df['Cluster'] = kmeans.labels_
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
print("Cluster Centers:")
print(pd.DataFrame(cluster_centers, columns=df.columns[1:-1]))  # Use only relevant columns
plt.scatter(df['Product1'], df['Product2'], c=df['Cluster'], cmap='viridis')
plt.xlabel('Product1')
plt.ylabel('Product2')
plt.title('K-means Clustering of Customers based on Purchase History')
plt.show()
