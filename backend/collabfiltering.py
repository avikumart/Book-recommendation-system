import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# define the user item matrix function that creates a user-item matrix from the sampled book ratings dataframe
def create_user_item_matrix(data):
    user_item_matrix = data.pivot_table(index='user_id', 
                                        columns='isbn', 
                                        values='book_rating').fillna(0)
    
    # create the user mapper function
    user_mapper = {user: i for i, user in enumerate(user_item_matrix.index)}
    item_mapper = {item: i for i, item in enumerate(user_item_matrix.columns)}
    return user_item_matrix, user_mapper, item_mapper

# defin the class for the item based collaborative filtering
class ItemBasedCF:
    def __init__(self, data):
        self.user_item_matrix, self.user_mapper, self.item_mapper = create_user_item_matrix(data)
        self.item_user_matrix = self.user_item_matrix.T
        self.similarity_matrix = cosine_similarity(self.item_user_matrix)
        self.similarity_df = pd.DataFrame(self.similarity_matrix, 
                                          index=self.item_user_matrix.index, 
                                          columns=self.item_user_matrix.index)

    def get_similar_items(self, item_id, n=10):
        if item_id not in self.similarity_df.index:
            return []
        similar_items = self.similarity_df[item_id].sort_values(ascending=False).head(n + 1).index[1:]
        return similar_items.tolist()

    def recommend_items(self, user_id, n_recommendations=10):
        if user_id not in self.user_mapper:
            return []
        
        user_index = self.user_mapper[user_id]
        user_ratings = self.user_item_matrix.iloc[user_index]
        rated_items = user_ratings[user_ratings > 0].index.tolist()
        
        scores = {}
        for item in rated_items:
            similar_items = self.get_similar_items(item, n=20)
            for sim_item in similar_items:
                if sim_item not in rated_items:
                    if sim_item not in scores:
                        scores[sim_item] = 0
                    scores[sim_item] += self.similarity_df.at[item, sim_item] * user_ratings[item]
        
        ranked_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        recommended_items = [item for item, score in ranked_items[:n_recommendations]]
        
        return recommended_items