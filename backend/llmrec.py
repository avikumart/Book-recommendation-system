# write a function to get book recommendations from OpenAI API
import openai  
def get_book_recommendations(prompt, api_key, model="gpt-4", max_tokens=150):
    openai.api_key = api_key
    response = openai.ChatCompletion.create(
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
    recommendations = response.choices[0].message['content'].strip()
    return recommendations

# Example usage:
# api_key = "your_openai_api_key"
# prompt = "Can you recommend some science fiction books similar to Dune?"
# print(get_book_recommendations(prompt, api_key))