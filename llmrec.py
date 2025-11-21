# write a function to get book recommendations from OpenAI API
import openai
import streamlit as st

def get_book_recommendations(prompt, model="gpt-3.5-turbo", max_tokens=150):
    client = openai.OpenAI(api_key=st.session_state["openai_api"])
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides book recommendations based on user input and given titles of the book."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        n=1,
        stop=None,
        temperature=0.7,
    )
    recommendations = response.choices[0].message.content.strip()
    return recommendations

# write a function to rerank the recoomendation based llm api call and return the ranked list of recommendations based on the relevance to the input book title
def rerank_recommendations(recommendations, input_title, model="gpt-3.5-turbo", max_tokens=150):
    client = openai.OpenAI(api_key=st.session_state["openai_api"])
    prompt = f"Given the input book title '{input_title}', please rank the following recommendations based on their relevance to the input title:\n\n{recommendations}. give only the ranked list of recommendations without any explanation or additional text. it should in square brackets and each recommendation should be on a new line."
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that ranks book recommendations based on their relevance to the input book title."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        n=1,
        stop=None,
        temperature=0.7,
    )
    ranked_recommendations = response.choices[0].message.content.strip()
    return ranked_recommendations.split('\n')

# Example usage:
# api_key = "your_openai_api_key"
# prompt = "Can you recommend some science fiction books similar to Dune?"
# print(get_book_recommendations(prompt, api_key))