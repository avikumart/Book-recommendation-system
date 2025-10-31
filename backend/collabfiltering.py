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

# create book author user matrix function and author user based collaborative filtering class
def create_book_author_user_matrix(data):
    book_author_user_matrix = data.pivot_table(index='user_id', 
                                               columns='book_author', 
                                               values='book_rating').fillna(0)
    
    user_mapper = {user: i for i, user in enumerate(book_author_user_matrix.index)}
    author_mapper = {author: i for i, author in enumerate(book_author_user_matrix.columns)}
    return book_author_user_matrix, user_mapper, author_mapper

class AuthorBasedCF:
    def __init__(self, data):
        self.book_author_user_matrix, self.user_mapper, self.author_mapper = create_book_author_user_matrix(data)
        self.author_user_matrix = self.book_author_user_matrix.T
        self.similarity_matrix = cosine_similarity(self.author_user_matrix)
        self.similarity_df = pd.DataFrame(self.similarity_matrix, 
                                          index=self.author_user_matrix.index, 
                                          columns=self.author_user_matrix.index)

    def get_similar_authors(self, author_name, n=10):
        if author_name not in self.similarity_df.index:
            return []
        similar_authors = self.similarity_df[author_name].sort_values(ascending=False).head(n + 1).index[1:]
        return similar_authors.tolist()

    def recommend_authors(self, user_id, n_recommendations=10):
        if user_id not in self.user_mapper:
            return []
        
        user_index = self.user_mapper[user_id]
        user_ratings = self.book_author_user_matrix.iloc[user_index]
        rated_authors = user_ratings[user_ratings > 0].index.tolist()
        
        scores = {}
        for author in rated_authors:
            similar_authors = self.get_similar_authors(author, n=20)
            for sim_author in similar_authors:
                if sim_author not in rated_authors:
                    if sim_author not in scores:
                        scores[sim_author] = 0
                    scores[sim_author] += self.similarity_df.at[author, sim_author] * user_ratings[author]
        
        ranked_authors = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        recommended_authors = [author for author, score in ranked_authors[:n_recommendations]]
        
        return recommended_authors
    

# create book publisher user matrix function and publisher user based collaborative filtering class
def create_book_publisher_user_matrix(data):
    book_publisher_user_matrix = data.pivot_table(index='user_id', 
                                                  columns='book_publisher', 
                                                  values='book_rating').fillna(0)
    
    user_mapper = {user: i for i, user in enumerate(book_publisher_user_matrix.index)}
    publisher_mapper = {publisher: i for i, publisher in enumerate(book_publisher_user_matrix.columns)}
    return book_publisher_user_matrix, user_mapper, publisher_mapper

class PublisherBasedCF:
    def __init__(self, data):
        self.book_publisher_user_matrix, self.user_mapper, self.publisher_mapper = create_book_publisher_user_matrix(data)
        self.publisher_user_matrix = self.book_publisher_user_matrix.T
        self.similarity_matrix = cosine_similarity(self.publisher_user_matrix)
        self.similarity_df = pd.DataFrame(self.similarity_matrix, 
                                          index=self.publisher_user_matrix.index, 
                                          columns=self.publisher_user_matrix.index)

    def get_similar_publishers(self, publisher_name, n=10):
        if publisher_name not in self.similarity_df.index:
            return []
        similar_publishers = self.similarity_df[publisher_name].sort_values(ascending=False).head(n + 1).index[1:]
        return similar_publishers.tolist()

    def recommend_publishers(self, user_id, n_recommendations=10):
        if user_id not in self.user_mapper:
            return []
        
        user_index = self.user_mapper[user_id]
        user_ratings = self.book_publisher_user_matrix.iloc[user_index]
        rated_publishers = user_ratings[user_ratings > 0].index.tolist()
        
        scores = {}
        for publisher in rated_publishers:
            similar_publishers = self.get_similar_publishers(publisher, n=20)
            for sim_publisher in similar_publishers:
                if sim_publisher not in rated_publishers:
                    if sim_publisher not in scores:
                        scores[sim_publisher] = 0
                    scores[sim_publisher] += self.similarity_df.at[publisher, sim_publisher] * user_ratings[publisher]
        
        ranked_publishers = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        recommended_publishers = [publisher for publisher, score in ranked_publishers[:n_recommendations]]
        
        return recommended_publishers