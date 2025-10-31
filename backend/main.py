# import fastapi modules to create the API for the recommendation system using collaborative filtering
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import openai  
import os
# Load environment variables for OpenAI API key
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# import the collaborative filtering classes and functions
from backend.collabfiltering import ItemBasedCF, create_user_item_matrix

# Initialize FastAPI app
app = FastAPI()

# pydantics models for request and response bodies
class ItemRatings(BaseModel):
    isbn: int
    ratings: Dict[int, float]  # item_id: rating

class RecommendedItems(BaseModel):
    isbn: int
    recommended_items: List[int]

# pydantic models for request and response bodies
class BookTitleRequest(BaseModel):
    title: str

class BookRecommendationResponse(BaseModel):
    title: str
    recommendations: List[str] 

# load sampled book ratings data for the recommendation system
data = pd.read_csv('data/sampled_book_ratings.csv')

# create user-item matrix
user_item_matrix, user_map, item_map = create_user_item_matrix(data)

# create item-based collaborative filtering model
item_cf = ItemBasedCF(user_item_matrix)

# health check endpoint
@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/recommend", response_model=RecommendedItems)
def recommend_items(item_ratings: ItemRatings):
    try:
        recommended_items = item_cf.recommend_items(item_ratings.isbn, n_recommendations=10)
        return RecommendedItems(isbn=item_ratings.isbn, recommended_items=recommended_items)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# llm call to to generate recommendations based on book title given by user
@app.post("/llm_recommend", response_model=BookRecommendationResponse)
def llm_recommend_books(request: BookTitleRequest):
    try:
        prompt = f"Recommend 5 books similar to the book titled '{request.title}'. Provide only the book titles in a list format."
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.7,
        )
        recommendations_text = response.choices[0].text.strip()
        recommendations = [title.strip() for title in recommendations_text.split('\n') if title.strip()]
        return BookRecommendationResponse(title=request.title, recommendations=recommendations)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))