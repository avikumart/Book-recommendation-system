import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import sys
import os

# --- MOCKING STREAMLIT & DEPENDENCIES BEFORE IMPORT ---
# We mock these modules specifically so that importing 'app.py' doesn't 
# try to run the actual Streamlit UI command (like st.set_page_config) 
# or connect to OpenAI during the test discovery phase.
sys.modules['streamlit'] = MagicMock()
sys.modules['openai'] = MagicMock()
sys.modules['clustering'] = MagicMock()
sys.modules['collabfiltering'] = MagicMock()
sys.modules['llmrec'] = MagicMock()
sys.modules['helpers'] = MagicMock()

# Now we can import the app. 
# Note: We might need to handle the specific functions if they rely on the mocks.
import app

class TestBookRecommendationApp(unittest.TestCase):

    def setUp(self):
        # Reset the mock for streamlit error before each test
        app.st.error.reset_mock()

    # --- TESTS FOR combined_recommendations ---

    def test_combined_recommendations_success(self):
        """Test merging collaborative filtering and LLM recommendations."""
        rec_titles = ["Book A", "Book B"]
        llm_recs = ["Book C", "Book D"]
        isbn = "12345"

        result = app.combined_recommendations(rec_titles, llm_recs, isbn)
        
        expected = ["Book A", "Book B", "Book C", "Book D"]
        self.assertEqual(result, expected)

    def test_combined_recommendations_empty_inputs(self):
        """Test logic when input lists are empty."""
        rec_titles = []
        llm_recs = []
        isbn = "12345"

        result = app.combined_recommendations(rec_titles, llm_recs, isbn)
        self.assertEqual(result, [])

    def test_combined_recommendations_handles_exception(self):
        """Test that the function handles errors gracefully (returns empty list)."""
        # Pass None to trigger a likely TypeError when trying to concatenate
        result = app.combined_recommendations(None, ["Book A"], "12345")
        
        # Expecting empty list and st.error to be called
        self.assertEqual(result, [])
        app.st.error.assert_called_once()
        self.assertIn("Error getting combined recommendations", app.st.error.call_args[0][0])

    # --- TESTS FOR load_data ---

    @patch('app.pd.read_csv')
    def test_load_data_success(self, mock_read_csv):
        """Test loading data successfully returns a DataFrame."""
        # Setup mock return value
        mock_df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        mock_read_csv.return_value = mock_df
        
        file_path = "dummy_path.csv"
        result = app.load_data(file_path)
        
        pd.testing.assert_frame_equal(result, mock_df)
        mock_read_csv.assert_called_with(file_path)

    @patch('app.pd.read_csv')
    def test_load_data_failure(self, mock_read_csv):
        """Test that load_data returns empty DataFrame on error."""
        # Setup mock to raise exception
        mock_read_csv.side_effect = FileNotFoundError("File not found")
        
        file_path = "bad_path.csv"
        result = app.load_data(file_path)
        
        # Check that result is an empty DataFrame
        self.assertTrue(result.empty)
        # Check that st.error was called
        app.st.error.assert_called_with("Error loading data: File not found")

    # --- TESTS FOR load_sampled_data ---

    @patch('app.pd.read_csv')
    def test_load_sampled_data_success(self, mock_read_csv):
        """Test loading sampled data successfully."""
        mock_df = pd.DataFrame({'isbn': ['111', '222'], 'title': ['A', 'B']})
        mock_read_csv.return_value = mock_df
        
        file_path = "dummy_sampled.csv"
        result = app.load_sampled_data(file_path)
        
        pd.testing.assert_frame_equal(result, mock_df)

    @patch('app.pd.read_csv')
    def test_load_sampled_data_failure(self, mock_read_csv):
        """Test loading sampled data failure."""
        mock_read_csv.side_effect = Exception("Generic error")
        
        file_path = "dummy_sampled.csv"
        result = app.load_sampled_data(file_path)
        
        self.assertTrue(result.empty)
        app.st.error.assert_called_with("Error loading sampled data: Generic error")

if __name__ == '__main__':
    unittest.main()