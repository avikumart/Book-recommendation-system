# Book-recommendation-system
A hybrid book recommendation system that leverages collaborative filtering with the large language models to recommend highly relevant book list to the user

### Brief description of the project

Identifying the right books and information for academic or personal growth purposes can be a daunting task for many users, particularly traditional search engines, which rely on keyword searches. These search engines might produce irrelevant results instead of relevant books that are in the interests of the students. 

To eliminate the unnecessary search results, this project introduces an AI-powered recommender system that takes users' interests into account by expanding the query based on the search query using generative large language models and retrieves relevant books with authors and publication information from the databases. The system combines content-based filtering using item-user matrices, text clustering techniques and generative AI models to produce accurate and precise book recommendations to the users.

### Repository structure

/book-recommedataion-system
├── environment.yml
|-- .gitignore
|-- LICENSE
├── README.md
|-- /data
|-- /nbs
├── /backend
│   ├── main.py             # FastAPI application entry point, defines endpoints
│   ├── collabfiltering.py     # Contains collaborative filtering logic
│   ├── llmrec.py     
└── /frontend
    └── app.py    # Streamlit application (the user interface)

 
### System diagram

### Benefits of the system

A book recommendation system can have many benefits for the students and users alike. Here are the motivations and benefits of the project:

- To streamline the book search and recommendation using the latest AI models 
- Help students to ease the book search and retrieve accurate and precise book recommendations for certain keywords
- Demonstration of the traditional search techniques with the latest generative AI models 
- Leverage generative AI model capabilities to describe book content and expand user queries to many keywords
- Demonstrate the use case of the hybrid retrieval recommendation systems

### Proposed Technical Solution

- User interface to take the book query keywords as input 
- User query keywords will be fed to the API call to generate relevant new book titles or themes that align with the user query
- Book query will find similar items from the database based on the item similarity algorithm
- Generative AI-generated keywords will also retrieve relevant book titles or similar items from the database to find similar items to the user query
- Combined retrieved results from both functions will then be  clustered into groups to generate a detailed description of each recommendation using a large language model API call
- Finally, the retrieved results will be presented to the user on the user interface
- Results can be evaluated based on the Precision-Recall curve, RMSE, MAP and relevance judgment scoring, such as discounted cumulative gains, to ensure the system is validated for real-world use cases.
- For the book recommendation system, tools such as OpenAI/HuggingFace, Streamlit for the UI, FastAPI, the scikit-learn toolkit, LangChain, pandas, numpy, and the Matplotlib visualisation toolkit will be used.


### Tools and techniques used

### Setting up the project

### Contributing to the project

### Acknowledgements
