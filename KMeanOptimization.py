import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import numpy as np
from sklearn.decomposition import TruncatedSVD
import random
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from joblib import Parallel, delayed



"""
First, users are grouped based on similar rating behaviors using K-Means.
Then, for a target user, the system finds similar users within the same cluster and recommends movies based on their ratings.
"""
#data path benim bilgisayara göre
data_path = 'C:/Users/burcu/.cache/kagglehub/datasets/grouplens/movielens-20m-dataset/versions/1'
# Load data
ratings = pd.read_csv(f'{data_path}/rating.csv') # Make sure this file is available
#bunlar test için
print(ratings.head())
print(ratings.dtypes)
print(ratings.describe())

# Sample a subset of users or ratings(böyle yapmazsak memory yetmiyor hata alıyoruz dataset çok büyük çünkü)
sampled_users = ratings['userId'].drop_duplicates().sample(n=1000, random_state=42)#random olarak 1000 farklı id seçiyor
ratings_subset = ratings[ratings['userId'].isin(sampled_users)]#seçilmiş userlara ait ratingleri alıyor

# Now pivot this subset
user_item_matrix = ratings_subset.pivot(index='userId', columns='movieId', values='rating').fillna(0)
# Convert to sparse matrix and reduce dimensionality
data = csr_matrix(user_item_matrix.drop(columns='cluster', errors='ignore'))
svd = TruncatedSVD(n_components=50, random_state=42)
reduced_data = svd.fit_transform(data)

# Fit GMM with k=5
gmm = GaussianMixture(n_components=3, covariance_type='diag', random_state=42)
clusters = gmm.fit_predict(reduced_data)

# Add clusters to the original dataframe
user_item_matrix['cluster'] = clusters

# Optional: Visualize the clusters (e.g., using a 2D projection)
svd_2d = TruncatedSVD(n_components=2, random_state=42)
data_2d = svd_2d.fit_transform(data)
plt.scatter(data_2d[:, 0], data_2d[:, 1], c=clusters, cmap='viridis')
plt.title('User Clusters (k=3)')
plt.xlabel('SVD Component 1')
plt.ylabel('SVD Component 2')
plt.show()

data = csr_matrix(user_item_matrix.drop(columns='cluster', errors='ignore'))

# Reduce dimensionality to 50 components to handle ~3,900 movies
svd = TruncatedSVD(n_components=50, random_state=42)
reduced_data = svd.fit_transform(data)

# Narrow the range of k based on prior elbow (k=5) and silhouette (k=2-3) insights
K = range(2, 10)

# Parallel computation of GMM fits with optimized parameters
def fit_gmm(k, data):
    gmm = GaussianMixture(n_components=k, covariance_type='diag', max_iter=30, n_init=1, random_state=42)
    gmm.fit(data)
    return gmm.aic(data), gmm.bic(data)

results = Parallel(n_jobs=-1)(delayed(fit_gmm)(k, reduced_data) for k in K)
aic_scores, bic_scores = zip(*results)

# Plot AIC and BIC
plt.figure(figsize=(8, 5))
plt.plot(K, aic_scores, 'bx-', label='AIC')
plt.plot(K, bic_scores, 'ro-', label='BIC')
plt.xlabel('k')
plt.ylabel('Score')
plt.title('AIC and BIC For Different k')
plt.legend()
plt.show()

# Find the k with minimum AIC and BIC
optimal_k_aic = K[np.argmin(aic_scores)]
optimal_k_bic = K[np.argmin(bic_scores)]
print(f"Optimal k based on AIC: {optimal_k_aic}")
print(f"Optimal k based on BIC: {optimal_k_bic}")


distortions = []
K = range(2, 30)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(user_item_matrix.drop(columns='cluster', errors='ignore'))
    distortions.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion (Inertia)')
plt.title('Elbow Method For Optimal k')
plt.legend()  # Add this line
plt.show()


silhouette_scores = []
K = range(2, 30)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(user_item_matrix.drop(columns='cluster', errors='ignore'))
    silhouette_avg = silhouette_score(user_item_matrix.drop(columns='cluster', errors='ignore'), cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Plot silhouette scores
plt.figure(figsize=(8, 5))
plt.plot(K, silhouette_scores, 'bo-')
plt.xlabel('k')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score For Different k')
plt.legend()
plt.show()

#Bütün testler ayrı ayrı çalıştırıldı, tekrar denemek için comment out yapılmalı

