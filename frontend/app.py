import streamlit as st

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

# create the user interface for the book recommendation system to take user input of the book name
st.header("Get Book Recommendations")
book_name = st.text_input("Enter a book name you like:")
if st.button("Get Recommendations"):
    if book_name:
        # Placeholder for recommendation logic
        st.write(f"Recommendations for '{book_name}':")
        # Here you would call your recommendation function and display results
        st.write("1. Book A\n2. Book B\n3. Book C")  # Example output
    else:
        st.error("Please enter a book name to get recommendations.")    