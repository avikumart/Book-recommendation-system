from surprise import SVD
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split
from collections import defaultdict
import pickle
from sklearn.metrics import mean_squared_error
import pandas as pd

class SVDRecommender:
    def __init__(self, n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02):
        self.model = SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all)
        self.trained = False

    def fit(self, data):
        # Load data into Surprise format
        reader = Dataset.load_from_df(data[['user_id', 'isbn', 'book_rating']], reader=None)
        trainset, self.testset = train_test_split(reader, test_size=0.2)
        
        # Train the SVD model
        self.model.fit(trainset)
        self.trained = True

    def predict(self, user_id, item_id):
        if not self.trained:
            raise Exception("Model not trained yet. Call fit() before predict().")
        return self.model.predict(user_id, item_id).est

    def evaluate(self):
        if not self.trained:
            raise Exception("Model not trained yet. Call fit() before evaluate().")
        predictions = self.model.test(self.testset)
        rmse = accuracy.rmse(predictions)
        return rmse

    def save_model(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)

    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        self.trained = True


# Write a function to get top-N recommendations for all users
def get_top_n_recommendations(model, data, n=10):
    if not model.trained:
        raise Exception("Model not trained yet. Call fit() before getting recommendations.")
    
    # Get all unique users and items
    users = data['user_id'].unique()
    items = data['isbn'].unique()
    
    # Create a dictionary to hold the top-N recommendations for each user
    top_n = defaultdict(list)
    
    for user in users:
        # Predict ratings for all items for the current user
        predictions = [model.predict(user, item) for item in items]
        
        # Sort predictions by estimated rating in descending order
        predictions.sort(key=lambda x: x.est, reverse=True)
        
        # Get the top-N item ids
        top_n[user] = [pred.iid for pred in predictions[:n]]
    
    return top_n