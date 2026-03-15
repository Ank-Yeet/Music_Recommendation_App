import streamlit as st

st.set_page_config(
    page_title="Music Recommender 🎵",
    page_icon="🎧",
    layout="centered"
)

st.title("🎶 Instant Music Recommender")

# ---- Safe backend import ----
try:
    from recommend import df, recommend_songs
except Exception as e:
    st.error("❌ Failed to load recommendation engine")
    st.exception(e)
    st.stop()

# ---- UI ----
song_list = sorted(df['song'].dropna().unique())
selected_song = st.selectbox("🎵 Select a song:", song_list)

if st.button("🚀 Recommend Similar Songs"):
    with st.spinner("Finding similar songs..."):
        recommendations = recommend_songs(selected_song)

    if recommendations is None:
        st.warning("Song not found.")
    else:
        st.success("Top similar songs:")
        st.dataframe(recommendations, use_container_width=True)
