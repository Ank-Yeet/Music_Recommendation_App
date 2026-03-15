import pandas as pd
import re
import nltk
import joblib
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(__file__)
CSV_PATH = os.path.join(BASE_DIR, "spotify_millsongdata.csv")

print("🚀 Starting preprocessing...")

df = pd.read_csv(CSV_PATH).sample(10000)
df = df.drop(columns=['link'], errors='ignore').reset_index(drop=True)

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", str(text))
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

df['cleaned_text'] = df['text'].apply(preprocess_text)

tfidf = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['cleaned_text'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

joblib.dump(df, os.path.join(BASE_DIR, 'df_cleaned.pkl'))
joblib.dump(cosine_sim, os.path.join(BASE_DIR, 'cosine_sim.pkl'))

print("✅ PKL files created successfully")
