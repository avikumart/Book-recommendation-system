from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Import your collaborative filtering modules
# Ensure collabfiltering.py exists in the same directory
from collabfiltering import ItemBasedCF, create_user_item_matrix

# Initialize FastAPI app
app = FastAPI()

# --- Pydantic Models ---
class ItemRatings(BaseModel):
    isbn: str

class RecommendedItems(BaseModel):
    item_book_titles: List[str] 

class BookTitleRequest(BaseModel):
    titles: List[str]

class BookRecommendationResponse(BaseModel):
    titles: List[str]
    recommendations: List[str] 

# --- Data Loading ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, 'data', 'user_item_matrix.csv')

def load_data(file_path: str) -> pd.DataFrame:
    # Check if file exists first to avoid crashing
    if not os.path.exists(file_path):
        print(f"WARNING: Data file not found at {file_path}")
        # Return empty DF with required columns to prevent KeyError downstream
        return pd.DataFrame(columns=['isbn', 'book_title'])
        
    try:
        book_ratings = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        print(f"Data shape: {book_ratings.shape}")
        return book_ratings
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame(columns=['isbn', 'book_title'])

# Load the data
data = load_data(file_path)

# Initialize logic only if data exists
if not data.empty:
    user_map, item_map = create_user_item_matrix(data)
    item_cf = ItemBasedCF(data)
else:
    print("WARNING: DataFrame is empty. Recommendation endpoints will fail.")
    item_cf = None

# --- Endpoints ---
@app.get("/health")
def health_check():
    return {"status": "ok"}

# api endpoint to get item-based collaborative filtering recommendations
@app.post("/recommend", response_model=RecommendedItems)
def recommend_items(item_ratings: ItemRatings):
    if item_cf is None:
        raise HTTPException(status_code=503, detail="Recommendation model not loaded (Data missing).")
        
    try:
        recommended_items = item_cf.get_similar_items(item_ratings.isbn, n=5)
        # Check if ISBNs exist in the data before filtering
        recommended_titles = data[data['isbn'].isin(recommended_items)]['book_title'].tolist()
        
        return RecommendedItems(
                item_book_titles=recommended_titles
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# api endpoint to get LLM-based book recommendations
@app.post("/llm_recommend", response_model=BookRecommendationResponse)
def llm_recommend_books(request: BookTitleRequest):
    try:
        titles_list = "', '".join(request.titles)
        prompt = f"""Recommend 5 books similar to the book titled '{titles_list}'. 
        Provide only the book titles in a list format. Make sure to include only the book titles without any additional information.
        """
        
        client = openai.OpenAI()
        
        # FIXED: Use 'messages' for Chat models, not 'prompt'
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful book recommendation assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            n=1,
            temperature=0.7,
        )
        
        # FIXED: Access content via object attributes, not dictionary keys
        recommendations_text = response.choices[0].message.content.strip()
        
        recommendations = [title.strip() for title in recommendations_text.split('\n') if title.strip()]
        
        return BookRecommendationResponse(titles=request.titles, recommendations=recommendations)
    except Exception as e:
        # Log the error to console so you can see it in terminal
        print(f"LLM Error: {e}") 
        raise HTTPException(status_code=500, detail=f"LLM processing error: {str(e)}")