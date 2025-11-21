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
from llmrec import get_book_recommendations, rerank_recommendations
from helpers import clean_recommendations

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
    st.session_state["openai_api"] = api_key
    st.sidebar.success("API Key set successfully!")
else:
    st.sidebar.warning("Please enter your API Key to proceed.")

# write the load data function as per the main.py file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, 'data', 'user_item_matrix.csv')

@st.cache_data  
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
@st.cache_data
def load_sampled_data(file_path: str) -> pd.DataFrame:
    try:
        sampled_data = pd.read_csv(file_path)
        print(f"Sampled data loaded successfully from {file_path}")
        print(f"Sampled data shape: {sampled_data.shape}")
        return sampled_data
    except Exception as e:
        st.error(f"Error loading sampled data: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of error
    
file_path_sampled = os.path.join(BASE_DIR, 'data', 'sampled_book_ratings.csv')
sampled_data = load_sampled_data(file_path_sampled)

recommended_titles = []
recommendations = []    

# streamlit front end to call the recommend_items function from the main.py file and display the results
if isbn:
    try:
        recommended_items = item_cf.get_similar_items(isbn, n=5)
        recommended_titles = sampled_data[sampled_data['isbn'].isin(recommended_items)]['book_title'].tolist()
        input_title = sampled_data[sampled_data['isbn'] == isbn]['book_title'].iloc[0]
        st.subheader("Recommended Books:")
        # wrap the recommneded titles in streamlit container of expanders to show the recommended books in a collapsible format
        with st.expander("Click to see recommended books"):
            # call the rerank_recommendations function to get the ranked list of recommendations based on the relevance to the input book title
            ranked_recommendations = rerank_recommendations(recommended_titles, input_title)
            ranked_recommendations = [rec.strip() for rec in ranked_recommendations if rec.strip()]
            st.write(f"Ranked recommendations for '{input_title}':")
            for _, title in enumerate(ranked_recommendations, 1):
                st.write(f"{title}")
    except Exception as e:
        st.error(f"Error getting recommendations: {e}")

# streamlit front end to call the llm recommendation function from the main.py file and display the results
if isbn and "openai_api" in st.session_state:
    titles = sampled_data[sampled_data['isbn'] == isbn]['book_title'].tolist()
    if titles:
        st.write(f"Getting LLM-based recommendations for {', '.join([title for title in titles])}...")
        client = openai.OpenAI(api_key=st.session_state["openai_api"])
        response = client.chat.completions.create(model="gpt-3.5-turbo",
                messages=[
                {"role": "system", "content": "You are a helpful assistant that provides book recommendations based on user input."},
                {"role": "user", "content": f"Please recommend  books similar to the {', '.join([title for title in titles])}. only include the book titles with explanations of why they are similar to the input book titles. Please provide the recommendations in a numbered list format."}],
                max_tokens=150,
                n=1,
                stop=None,
                temperature=0.7,
                    )
        recommendations_text = response.choices[0].message.content.strip()
        if recommendations_text:
            st.subheader("LLM-based recommendations received:")
            with st.expander("Click to see LLM-based recommended books"):
                st.write(recommendations_text)
        recommendations = [title.strip() for title in recommendations_text.split('\n') if title.strip()]
        recommendations = clean_recommendations(recommendations)

# functions that combiine the collaborative filtering and llm recommendations to provide a more comprehensive recommendation list
def combined_recommendations(recommended_titles, recommendations, isbn) -> List[str]:
    try:
        combined_recs = []
        # collaborative filtering recommendations
        combined_recs.extend(recommended_titles + recommendations)
        return combined_recs
    except Exception as e:
        st.error(f"Error getting combined recommendations: {e}")
        return []
    
# display in streamlit frontend using dropdown container
if isbn:
    recs = combined_recommendations(recommended_titles, recommendations, isbn)
    if recs:
        st.subheader("Combined Recommended Books:")
        with st.expander("Click to see combined recommended books"):
            for _,book in enumerate(recs):
                st.write(f"{book}")
    else:
        st.write("No combined recommendations found.")

# cluster-based recommendations with streamlit frontend functions
if isbn and "openai_api" in st.session_state:
    recs = clustering.generate_recommendations(isbn, item_cf, sampled_data)
    if recs:
        st.subheader("User query based Recommended Books:")
        with st.expander("Click to see user query based recommended books"):
            st.write(recs)
    else:
        st.write("No user query based recommendations found.")
    cluster_recs = clustering.cluster_recommendations(recs)
    cluster_descriptions = clustering.generate_cluster_descriptions(cluster_recs)
    if cluster_descriptions:
        st.subheader("Cluster Descriptions:")
        with st.expander("Click to see cluster descriptions"):
            for label, item in cluster_descriptions.items():
                st.write(f"Cluster description of the label {label}:\n {item}")
    else:
        st.write("No cluster descriptions found.")