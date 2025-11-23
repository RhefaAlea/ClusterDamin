# K-MEANS CLUSTERING DATA COVID EXCEL

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Pre Proccessing

df = pd.read_excel("ClusterCovid.xlsx")

print("=== 5 DATA AWAL ===")
print(df.head())

X = df[['Confirmed', 'Deaths', 'Recovered']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


#Elbow Methods
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel("Jumlah Cluster (k)")
plt.ylabel("WCSS")
plt.title("Elbow Method untuk K-Means")
plt.grid(True)
plt.show()


#Running K-Means with k = 3
k = 2 
kmeans = KMeans(n_clusters=k, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

df['Cluster'] = cluster_labels



print("\n=== DATA HASIL CLUSTER ===")
print(df[['Province/State', 'Country/Region', 'Confirmed', 'Deaths', 'Recovered', 'Cluster']])

df.to_excel("hasil_cluster_covid.xlsx", index=False)
print("\nFile hasil disimpan sebagai: hasil_cluster_covid.xlsx")

