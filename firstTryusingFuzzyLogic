import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Sample Knowledge Base for BNS Sections
knowledge_base = {
    'Section 14': {
        'title': 'Possession of Firearms',
        'conditions': [
            'Illegal possession of firearms',
            'Possession of loaded firearms'
        ]
    },
    'Section 15': {
        'title': 'Theft of Firearms',
        'conditions': [
            'Theft of firearms',
            'Illegal sale of firearms'
        ]
    },
    'Section 16': {
        'title': 'Unlawful Use of Weapons',
        'conditions': [
            'Unlawful use of weapons',
            'Discharge of firearms in public'
        ]
    },
    'Section 22': {
        'title': 'Public Disturbance',
        'conditions': [
            'Public disturbance',
            'Disturbance in a public place'
        ]
    }
}

# Crime Report
crime_report = "
# Step 1: Preprocess the Crime Report
nltk_stopwords = set(stopwords.words('english'))

def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text.lower())
    # Remove stopwords and punctuation
    tokens = [word for word in tokens if word.isalpha() and word not in nltk_stopwords]
    return ' '.join(tokens)

processed_report = preprocess_text(crime_report)

# Step 2: Prepare Data for Similarity Calculation
# Gather conditions from each section
sections = []
conditions = []
for section, details in knowledge_base.items():
    for condition in details['conditions']:
        sections.append(section)
        conditions.append(condition)

# Add processed report as the last "document" in the list for similarity calculation
documents = conditions + [processed_report]

# Step 3: Calculate TF-IDF Vectors and Similarity
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Calculate similarity of the crime report to each section's conditions
similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]

# Map similarity scores to sections by averaging scores of multiple conditions
section_scores = {}
for i, section in enumerate(sections):
    if section not in section_scores:
        section_scores[section] = []
    section_scores[section].append(similarity_scores[i])

# Average similarity scores for each section
for section in section_scores:
    section_scores[section] = np.mean(section_scores[section])

# Step 4: Apply Fuzzy Logic to Determine Confidence Levels
def fuzzy_confidence(score, thL=0.3, thU=0.7):
    if score >= thU:
        return "High"
    elif score >= thL:
        return "Medium"
    else:
        return "Low"

confidence_levels = {section: fuzzy_confidence(score) for section, score in section_scores.items()}

# Prepare Results
results = []
for section, confidence in confidence_levels.items():
    results.append({
        'Section': section,
        'Title': knowledge_base[section]['title'],
        'Confidence Level': confidence,
        'Similarity Score': round(section_scores[section], 2)
    })

# Create DataFrame for readability and sort by confidence level
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by=['Confidence Level', 'Similarity Score'], ascending=[False, False])

print("Applicable IPC Sections:")
print(results_df)
