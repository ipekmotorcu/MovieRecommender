import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import numpy as np
import random
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

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

# bunlar test için
"""
print(movies.head())
print(ratings.head())
print(ratings.dtypes)
print(ratings.describe())
"""

# Sample a subset of users or ratings(böyle yapmazsak memory yetmiyor hata alıyoruz dataset çok büyük çünkü)
sampled_users = ratings['userId'].drop_duplicates().sample(n=1000,
                                                           random_state=42)  # random olarak 1000 farklı id seçiyor
ratings_subset = ratings[ratings['userId'].isin(sampled_users)]  # seçilmiş userlara ait ratingleri alıyor

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

#Farklı similarity methodları
def pearson_similarity(target_vector, other_vectors):
    similarities = []
    for vec in other_vectors:
        corr, _ = pearsonr(target_vector.flatten(), vec)
        similarities.append(corr if not np.isnan(corr) else 0)
    return np.array(similarities)


#it considers peoples harsh evaluation with subtracting each user's mean rating
def mean_centered_cosine_similarity(target_vector, other_vectors):
    # Ignore 0s (unrated) when computing means
    target_nonzero = target_vector != 0
    target_mean = target_vector[target_nonzero].mean() if np.any(target_nonzero) else 0
    target_centered = np.where(target_nonzero, target_vector - target_mean, 0)

    others_centered = []
    for vec in other_vectors:
        mask = vec != 0
        mean = vec[mask].mean() if np.any(mask) else 0
        centered_vec = np.where(mask, vec - mean, 0)
        others_centered.append(centered_vec)

    others_centered = np.vstack(others_centered)
    return cosine_similarity(target_centered.reshape(1, -1), others_centered)[0]


# Step 1: KMeans clustering
n_clusters = 10  # You can tune this BUNUN İÇİN ELBOW METHOD KULLANILABİLİR
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
user_clusters = kmeans.fit_predict(user_item_matrix)

# Assign cluster labels
user_item_matrix['cluster'] = user_clusters


# Step 2: Recommend for a given user based on neighbors in the same cluster
def recommend_movies(user_id, top_n=5, similarity_metric="cosine"):
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
        # Compute similarity
    if similarity_metric == "cosine":
        similarities = cosine_similarity(target_vector, cluster_users.values)[0]

    elif similarity_metric == "pearson":
        similarities = pearson_similarity(target_vector, cluster_users.values)


    elif similarity_metric == "mean_centered_cosine":
        similarities = mean_centered_cosine_similarity(target_vector.flatten(), cluster_users.values)

    else:
        raise ValueError("Invalid similarity metric")

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
# random bir user'ın test setindeki gerçekten beğendiği film isimlerini ve önerilen film isimlerini gösteriyor
random_user = random.choice(
    user_item_matrix.index.tolist())  # bir seed yok her run'da farklı bir user'a bakıyor iyi bir sonuç gelirse o id'yi unutmayalım
print(f"\n Random test user: {random_user}")

recommended_ids = recommend_movies(user_id=random_user, top_n=10)
recommended_titles = movies[movies['movieId'].isin(recommended_ids)][['movieId', 'title']]

actual_ids = test_df[test_df['userId'] == random_user]['movieId'].tolist()
actual_titles = movies[movies['movieId'].isin(actual_ids)][['movieId', 'title']]

# ikisinde de olan filmler(Hit olanlar aslında)
matched_ids = list(set(recommended_ids) & set(actual_ids))
matched_titles = movies[movies['movieId'].isin(matched_ids)][['movieId', 'title']]

print("\n Actual (Test Set) Movies:")
print(actual_titles)

print("\n Recommended Movies:")
print(recommended_titles)

print("\n Matched (Correctly Recommended) Movies:")
print(matched_titles if not matched_titles.empty else "None")


# Metrics

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

# TEST

def evaluate(similarity_metric):
    k_test = 50
    precision_scores = []
    recall_scores = []
    hit_scores = []
    test_users = test_df['userId'].unique()

    for user_id in test_users:
        actual = test_df[test_df['userId'] == user_id]['movieId'].tolist()
        predicted = recommend_movies(user_id, top_n=10, similarity_metric=similarity_metric)

        if predicted:
            precision_scores.append(precision_at_k(actual, predicted, k=k_test))
            recall_scores.append(recall_at_k(actual, predicted, k=k_test))
            hit_scores.append(hit_rate(actual, predicted))

    print(f"Results using {similarity_metric.upper()}:")
    print(f"Average Precision@10: {np.mean(precision_scores):.4f}")
    print(f"Average Recall@10:    {np.mean(recall_scores):.4f}")
    print(f"Hit Rate:             {np.mean(hit_scores):.4f}")
    print("-" * 40)


#run evaluate for all metrics
for metric in [ "mean_centered_cosine","cosine", "pearson"]:
    evaluate(metric)