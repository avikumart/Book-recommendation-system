# write a function to get book recommendations from OpenAI API
import openai
import streamlit as st

def get_book_recommendations(prompt, model="gpt-3.5-turbo", max_tokens=150):
    client = openai.OpenAI(api_key=st.session_state["openai_api"])
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides book recommendations."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        n=1,
        stop=None,
        temperature=0.7,
    )
    recommendations = response.choices[0].message.content.strip()
    return recommendations


# Example usage:
# api_key = "your_openai_api_key"
# prompt = "Can you recommend some science fiction books similar to Dune?"
# print(get_book_recommendations(prompt, api_key))