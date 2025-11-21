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