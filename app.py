from typing import List
import streamlit as st
import requests
import openai
import pandas as pd
import clustering
import os
from dotenv import load_dotenv

# import the collaborative filtering classes and functions
from collabfiltering import ItemBasedCF, create_user_item_matrix

# Set page configuration
st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide"
)

# set the title of the app
st.title("📚 Book Recommendation System")

# the main content of the app goes here
# create a sidebar for the API key input
st.sidebar.header("API Key Configuration")
api_key = st.sidebar.text_input("Enter your API Key:", type="password")
if api_key:
    st.session_state["openai_client"] = openai.OpenAI(api_key=api_key)
    st.sidebar.success("API Key and Client set successfully!")
else:
    st.sidebar.warning("Please enter your API Key to proceed.")

# write the load data function as per the main.py file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, 'data', 'user_item_matrix.csv')

def load_data(file_path: str) -> pd.DataFrame:
    try:
        book_ratings = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        print(f"Data shape: {book_ratings.shape}")
        return book_ratings
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of error

data = load_data(file_path) 
# create the user interface for the book recommendation system to take user input of the isbn number
st.header("Get Book Recommendations")
isbn = st.text_input("Enter an ISBN number you like:")
if st.button("Get Recommendations"):
    if isbn:
        # Placeholder for recommendation logic
        st.write(f"Recommendations for ISBN '{isbn}':")
    else:
        st.error("Please enter an ISBN number to get recommendations.")

# create user-item matrix
user_map, item_map = create_user_item_matrix(data)

# create item-based collaborative filtering model
item_cf = ItemBasedCF(data)

# load the sampled book ratings data
file_path_sampled = os.path.join(BASE_DIR, 'data', 'sampled_book_ratings.csv')
sampled_data = load_data(file_path_sampled)

# streamlit front end to call the recommend_items function from the main.py file and display the results
if isbn:
    try:
        recommended_items = item_cf.get_similar_items(int(isbn), n=5)
        recommended_titles = sampled_data[sampled_data['isbn'].isin(recommended_items)]['book_title'].tolist()
        st.subheader("Recommended Books:")
        for idx, title in enumerate(recommended_titles, 1):
            st.write(f"{idx}. {title}")
    except Exception as e:
        st.error(f"Error getting recommendations: {e}")

# streamlit front end to call the llm recommendation function from the main.py file and display the results
if isbn and "openai_client" in st.session_state:
    try:
        titles = sampled_data[sampled_data['isbn'] == int(isbn)]['book_title'].tolist()
        if not titles:
            st.error(f"No book found with ISBN {isbn}")
        else:
            isbn = int(isbn)
            st.write(f"Getting LLM-based recommendations for '{isbn}'...")
            client = st.session_state["openai_client"]
            response = client.chat.completions.create(model="gpt-3.5-turbo",
                messages=[
                {"role": "system", "content": "You are a helpful assistant that provides book recommendations based on user input."},
                {"role": "user", "content": f"Please recommend 5 books similar to the book {isbn}."}],
                max_tokens=150,
                n=1,
                stop=None,
                temperature=0.7,
                    )
        recommendations_text = response.choices[0].message.content.strip()
        if recommendations_text:
            st.write("LLM-based recommendations received:")
            st.write(recommendations_text)
        recommendations = [title.strip() for title in recommendations_text.split('\n') if title.strip()]
        st.subheader("LLM-based Recommended Books:")
        for idx, title in enumerate(recommendations, 1):
            st.write(f"{idx}. {title}")
    except Exception as e:
        st.error(f"Error getting LLM-based recommendations: {e}")

# functions that combiine the collaborative filtering and llm recommendations to provide a more comprehensive recommendation list
def combined_recommendations(recommended_titles, recommendations, isbn, n: int = 5) -> List[str]:
    print(f"Getting combined recommendations for ISBN: {isbn}")
    try:
        combined_recs = set()
        # Get item-based collaborative filtering recommendations
        combined_recs.update(recommended_titles + recommendations)
        return list(combined_recs)[:n]
    except Exception as e:
        st.error(f"Error getting combined recommendations: {e}")
        return []
    
# display in strmlit frontend using dropdown container
if isbn:
    recs = combined_recommendations(recommended_titles, recommendations, isbn, n=5)
    if recs:
        st.subheader("Combined Recommended Books:")
        for idx, book in enumerate(recs, 1):
            st.write(f"{idx}. {book}")
    else:
        st.write("No combined recommendations found.")

# cluster-based recommendations with streamlit frontend functions
if isbn and "openai_client" in st.session_state:
    recs = clustering.generate_recommendations(int(isbn), item_cf, sampled_data)
    if recs:
        st.subheader("User query based Recommended Books:")
        for idx, book in enumerate(recs, 1):
            st.write(f"{idx}. {book}")
    else:
        st.write("No user query based recommendations found.")
    cluster_recs = clustering.cluster_recommendations(recs)
    cluster_descriptions = clustering.generate_cluster_descriptions(cluster_recs)
    if cluster_descriptions:
        st.subheader("Cluster Descriptions:")
        for idx, desc in enumerate(cluster_descriptions, 1):
            st.write(f"{idx}. {desc}")
    else:
        st.write("No cluster descriptions found.")  