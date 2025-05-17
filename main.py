import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import numpy as np
import random
from sklearn.model_selection import train_test_split
"""
First, users are grouped based on similar rating behaviors using K-Means.
Then, for a target user, the system finds similar users within the same cluster and recommends movies based on their ratings.
"""
"""
Test-> user listesini test ve train  
Elbow method -> cluster and (AIC/BIC or Sqrt)

!!!!!!
"""

ratings = pd.read_csv('rating.csv')  
movies = pd.read_csv('movie.csv')


#bunlar test için
"""
print(movies.head())
print(ratings.head())
print(ratings.dtypes)
print(ratings.describe())
"""


# Sample a subset of users or ratings(böyle yapmazsak memory yetmiyor hata alıyoruz dataset çok büyük çünkü)
sampled_users = ratings['userId'].drop_duplicates().sample(n=1000, random_state=42)#random olarak 1000 farklı id seçiyor
ratings_subset = ratings[ratings['userId'].isin(sampled_users)]#seçilmiş userlara ait ratingleri alıyor

train_data = []
test_data = []


for user_id in ratings_subset['userId'].unique():
    user_ratings = ratings_subset[ratings_subset['userId'] == user_id]
    if len(user_ratings) >= 5:
        train, test = train_test_split(user_ratings, test_size=0.2, random_state=42)
        train_data.append(train)
        test_data.append(test)

train_df = pd.concat(train_data)
test_df = pd.concat(test_data)

# Build user-item matrix from train data
user_item_matrix = train_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)







# Step 1: KMeans clustering
n_clusters = 10  # You can tune this BUNUN İÇİN ELBOW METHOD KULLANILABİLİR
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
user_clusters = kmeans.fit_predict(user_item_matrix)

# Assign cluster labels
user_item_matrix['cluster'] = user_clusters

# Step 2: Recommend for a given user based on neighbors in the same cluster
def recommend_movies(user_id, top_n=5):
    if user_id not in user_item_matrix.index:
        return []

    user_cluster = user_item_matrix.loc[user_id, 'cluster']
    cluster_users = user_item_matrix[user_item_matrix['cluster'] == user_cluster].drop(columns='cluster')
    
    if user_id not in cluster_users.index:
        return []

    target_vector = cluster_users.loc[user_id].values.reshape(1, -1)
    cluster_users = cluster_users.drop(index=user_id)
    if cluster_users.empty:
        return []


    similarities = cosine_similarity(target_vector, cluster_users.values)[0]
    similar_users = cluster_users.index[np.argsort(similarities)[::-1]]

    k = 10
    neighbors = cluster_users.loc[similar_users[:k]]
    avg_ratings = neighbors.mean()

    user_rated = user_item_matrix.loc[user_id].drop('cluster')
    unrated_movies = user_rated[user_rated == 0].index
    recommended = avg_ratings.loc[unrated_movies].sort_values(ascending=False).head(top_n)

    return recommended.index.tolist()

# Example usage:Random bir user seçip ona film öneriyor listesine göre

"""
random_user = random.choice(user_item_matrix.index.tolist())
print("Random test user:", random_user)

recommended_movies = recommend_movies(user_id=random_user, top_n=10)
user_rated = user_item_matrix.loc[random_user].drop('cluster')
rated_movies= user_rated[user_rated == 1].index
print("Rated Movies:",rated_movies)
print("Recommended Movie IDs:", recommended_movies)
"""
#random bir user'ın test setindeki gerçekten beğendiği film isimlerini ve önerilen film isimlerini gösteriyor
random_user = random.choice(user_item_matrix.index.tolist())#bir seed yok her run'da farklı bir user'a bakıyor iyi bir sonuç gelirse o id'yi unutmayalım
print(f"\n Random test user: {random_user}")

recommended_ids = recommend_movies(user_id=random_user, top_n=10)
recommended_titles = movies[movies['movieId'].isin(recommended_ids)][['movieId', 'title']]

actual_ids = test_df[test_df['userId'] == random_user]['movieId'].tolist()
actual_titles = movies[movies['movieId'].isin(actual_ids)][['movieId', 'title']]

#ikisinde de olan filmler(Hit olanlar aslında)
matched_ids = list(set(recommended_ids) & set(actual_ids))
matched_titles = movies[movies['movieId'].isin(matched_ids)][['movieId', 'title']]


print("\n Actual (Test Set) Movies:")
print(actual_titles)

print("\n Recommended Movies:")
print(recommended_titles)

print("\n Matched (Correctly Recommended) Movies:")
print(matched_titles if not matched_titles.empty else "None")



#Metrics

def precision_at_k(actual, predicted, k):
    if not actual:
        return 0
    predicted_k = predicted[:k]
    hits = len(set(actual) & set(predicted_k))
    return hits / k

def recall_at_k(actual, predicted, k):
    if not actual:
        return 0
    predicted_k = predicted[:k]
    hits = len(set(actual) & set(predicted_k))
    return hits / len(actual)

def hit_rate(actual, predicted):
    return int(len(set(actual) & set(predicted)) > 0)

"""
If the intersection of actual(test set) and predicted sets is non-empty ( at least one match), we call it a hit → score = 1
"""

#TEST

k_test=50
precision_scores = []
recall_scores = []
hit_scores = []

test_users = test_df['userId'].unique()

for user_id in test_users:
    actual = test_df[test_df['userId'] == user_id]['movieId'].tolist()
    predicted = recommend_movies(user_id, top_n=k_test)

    if predicted:
        precision_scores.append(precision_at_k(actual, predicted, k=k_test))
        recall_scores.append(recall_at_k(actual, predicted, k=k_test))
        hit_scores.append(hit_rate(actual, predicted))

#Results
print(f"Average Precision@10: {np.mean(precision_scores):.4f}")
print(f"Average Recall@10:    {np.mean(recall_scores):.4f}")
print(f"Hit Rate:             {np.mean(hit_scores):.4f}")