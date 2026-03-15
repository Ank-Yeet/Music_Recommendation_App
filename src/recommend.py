import joblib
import os
import streamlit as st

# ---------- Paths ----------
BASE_DIR = os.path.dirname(__file__)
DF_PATH = os.path.join(BASE_DIR, "df_cleaned.pkl")
SIM_PATH = os.path.join(BASE_DIR, "cosine_sim.pkl")

# ---------- Cached loaders ----------
@st.cache_resource
def load_model():
    df = joblib.load(DF_PATH)
    cosine_sim = joblib.load(SIM_PATH)
    return df, cosine_sim

df, cosine_sim = load_model()

# ---------- Recommendation function ----------
def recommend_songs(song_name, top_n=5):
    idx = df[df['song'].str.lower() == song_name.lower()].index

    if len(idx) == 0:
        return None

    idx = idx[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]

    song_indices = [i[0] for i in sim_scores]

    result_df = df[['artist', 'song']].iloc[song_indices].reset_index(drop=True)
    result_df.index = result_df.index + 1
    result_df.index.name = "S.No."

    return result_df
