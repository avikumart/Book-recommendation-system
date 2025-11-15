import streamlit as st
import requests
import openai
import pandas as pd
import clustering

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
    st.sidebar.success("API Key set successfully!")
else:
    st.sidebar.warning("Please enter your API Key to proceed.")

# store the openai key in session state
if "openai_key" not in st.session_state:
    st.session_state["openai_key"] = api_key

openai.api_key = st.session_state["openai_key"]

# create the user interface for the book recommendation system to take user input of the isbn number
st.header("Get Book Recommendations")
isbn = st.text_input("Enter an ISBN number you like:")
if st.button("Get Recommendations"):
    if isbn:
        # Placeholder for recommendation logic
        st.write(f"Recommendations for ISBN '{isbn}':")
    else:
        st.error("Please enter an ISBN number to get recommendations.")

# streamlit front end to call backend API and display recommendations
if api_key and isbn:
    response = requests.post(
        "http://localhost:8000/recommend",
        json={"isbn": isbn, "api_key": api_key}
    )
    if response.status_code == 200:
        recommendations = response.json().get("recommendations", [])
        if recommendations:
            st.subheader("Recommended Books:")
            for idx, book in enumerate(recommendations, 1):
                st.write(f"{idx}. {book}")
        else:
            st.write("No recommendations found.")
    else:
        st.error("Failed to fetch recommendations from the backend.")

# llm recommend api call for the llm rec display
if api_key and isbn:
    response = requests.post(
        "http://localhost:8000/llm_recommend",
        json={"book_titles": recommendations, "api_key": api_key}
    )
    if response.status_code == 200:
        llm_recommendations = response.json().get("llm_recommendations", [])
        if llm_recommendations:
            st.subheader("LLM Recommended Books:")
            for idx, book in enumerate(llm_recommendations, 1):
                st.write(f"{idx}. {book}")
        else:
            st.write("No LLM recommendations found.")
    else:
        st.error("Failed to fetch LLM recommendations from the backend.")

# cluster-based recommendations with streamlit frontend functions
if isbn:
    recs = clustering.generate_recommendations(isbn)
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