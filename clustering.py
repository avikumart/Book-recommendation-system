from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from backend.llmrec import get_book_recommendations
from backend.collabfiltering import ItemBasedCF, create_user_item_matrix
import openai
import pandas as pd
import numpy as np
import os 

# open ai api key
openai.api_key = os.getenv("OPENAI_API_KEY")

# run the collaborative filtering model and llmrec function to generate recommendations which will be clustered
# load sampled book ratings data for the recommendation system
path = '/Users/avikumart/Documents/GitHub/Book-recommendation-system/data/'
file_path = os.path.join(path, 'sampled_book_ratings.csv')

# define function to load data
def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)
data = load_data(file_path)

# create user-item matrix
user_item_matrix, user_map, item_map = create_user_item_matrix(data)

# create item-based collaborative filtering model
item_cf = ItemBasedCF(user_item_matrix)

# write function that takes isbn as a input to generate recommendations from the collaborative filtering model and then from llm apis in sequential manner
def generate_recommendations(isbn):
    # Get recommendations from collaborative filtering model
    collab_recommendations = item_cf.get_similar_items(isbn, n=5)
    # Get book titles for collaborative filtering recommendations
    collab_titles = data[data['isbn'].isin(collab_recommendations)]['book_title'].tolist()
    # Get recommendations from LLM based on collaborative filtering recommendations
    llm_recommendations = get_book_recommendations(f"Recommend books similar to: {', '.join(collab_titles)}", openai.api_key)
    return llm_recommendations

def cluster_recommendations(llmrecs, n_clusters=2, random_state=42):
    """
    Cluster the generated LLM recommendations using tfidf vectorization and apply kmeans algorithm.
    """
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    # verify if the llmrecs is a list of strings
    if not all(isinstance(rec.strip(), str) for rec in llmrecs):
        raise ValueError("All recommendations must be strings.")
    else:
        X = vectorizer.fit_transform([rec.strip() for rec in llmrecs])

    model = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = model.fit_predict(X)

    # assign the labels to the recommendations
    clustered_recommendations = list(zip(llmrecs, labels))
    return clustered_recommendations

# function to generate descriptions for each cluster using llm api call
def generate_cluster_descriptions(clustered_recommendations):
    cluster_dict = {}
    for rec, label in clustered_recommendations:
        cluster_dict.setdefault(label, []).append(rec)

    cluster_descriptions = {}
    for label, recs in cluster_dict.items():
        prompt = f"Generate a brief description for the following books: {', '.join(recs)}"
        description = get_book_recommendations(prompt, openai.api_key)
        cluster_descriptions[label] = description

    return cluster_descriptions