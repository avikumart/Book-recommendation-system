from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from llmrec import get_book_recommendations
import openai
from typing import List, Tuple, Dict, Any
import pandas as pd
import numpy as np
import os 
import re
import streamlit as st 

# write function that takes isbn as a input to generate recommendations from the collaborative filtering model and then from llm apis in sequential manner
def generate_recommendations(isbn, item_cf, data):
    # Get recommendations from collaborative filtering model
    collab_recommendations = item_cf.get_similar_items(isbn, n=5)
    # Get book titles for collaborative filtering recommendations
    collab_titles = data[data['isbn'].isin(collab_recommendations)]['book_title'].tolist()
    # Get recommendations from LLM based on collaborative filtering recommendations
    llm_recommendations = get_book_recommendations(f"Recommend books similar to: {', '.join(collab_titles)} or for {isbn}")
    return llm_recommendations


# write and helper function to extrac the list of the books from the recommendaded text
def extract_book_list(recoommed_text: str) -> List[str]:
    pattern = re.compile(r"^\s*(?:\d+[\.\)]|\d+\s+)\s*(.*?)\s*$", re.MULTILINE)
    matches = pattern.findall(recoommed_text)
    cleaned_titles = [item.strip().strip('"').strip("'") for item in matches if item.strip()]
    return cleaned_titles

def cluster_recommendations(llmrecs, n_clusters=2, random_state=42):
    """
    Cluster the generated LLM recommendations using tfidf vectorization and apply kmeans algorithm.
    """
    # extract the book titles
    book_list = extract_book_list(llmrecs)
    print("book_list:", book_list)

    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    # verify if the llmrecs is a list of strings
    if not book_list:
        st.warning("Cannot cluster: The input book list is empty after parsing.")
        return []

    # 3. Adjust n_clusters if necessary
    num_samples = len(book_list)
    if num_samples < 2:
        st.warning(f"Cannot cluster {num_samples} book(s). Returning without clustering.")
        return [(book_list[0], 0)] if book_list else []
        
    effective_n_clusters = min(n_clusters, num_samples)
    
    if effective_n_clusters < n_clusters:
        st.info(f"Reducing target clusters from {n_clusters} to {effective_n_clusters} because there are only {num_samples} samples.")

    try:
        # 4. Vectorization
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        X = vectorizer.fit_transform(book_list)

        # 5. Clustering
        model = KMeans(n_clusters=effective_n_clusters, random_state=random_state, n_init='auto')
        labels = model.fit_predict(X)

        # 6. Assign Labels
        clustered_recommendations = list(zip(book_list, labels))
        print("cluster_Rec:",clustered_recommendations)
        return clustered_recommendations
    
    except Exception as e:
        st.error(f"An error occurred during clustering: {e}")
        return []
    
# function to generate descriptions for each cluster using llm api call
def generate_cluster_descriptions(clustered_recommendations):
    cluster_dict = {}
    for rec, label in clustered_recommendations:
        cluster_dict.setdefault(label, []).append(rec)

    cluster_descriptions = {}
    for label, recs in cluster_dict.items():
        prompt = f"Generate a brief description for the following books: {', '.join(recs)}"
        description = get_book_recommendations(prompt)
        cluster_descriptions[label] = description
    print("cluster_distr:", cluster_descriptions)

    return cluster_descriptions