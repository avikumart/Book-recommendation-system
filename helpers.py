import re
import os

# a helper function to remove extra spaces and other texts from the recommended book titles including the numbering
def clean_recommendations(recommendations):
    cleaned_recommendations = []
    for rec in recommendations:
        # Remove numbering and extra spaces using regex
        cleaned_rec = re.sub(r'^\d+\.\s*', '', rec).strip()
        cleaned_recommendations.append(cleaned_rec)
    return cleaned_recommendations

# a helper function to take user feedback and save it as a json file in feedback folder
def save_feedback_for_ratings(isbn, feedback, feedback_folder):
    feedback_file = os.path.join(feedback_folder, 'feedback.json')    
    with open(feedback_file, 'a') as f:
        import json
        json.dump(feedback, f, indent=4)

def save_feedback(isbn, feedback, feedback_folder):
    feedback_file = os.path.join(feedback_folder, 'feedback.json')
    feedback_data = {
        "isbn": isbn,
        "feedback": feedback
    }
    with open(feedback_file, 'a') as f:
        import json
        json.dump(feedback_data, f, indent=4)