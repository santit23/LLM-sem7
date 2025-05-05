# streamlit_app.py
import streamlit as st
from tfidf_utils import get_embedding, get_nearest_neighbor

st.title("TF-IDF Word Embedding Explorer")

word = st.text_input("Enter a word:")

if word:
    embedding = get_embedding(word)
    if embedding is not None:
        st.success(f"Embedding found for '{word}'.")
        neighbors = get_nearest_neighbor(word)
        st.write("Top 5 nearest neighbors:")
        for w, score in neighbors:
            st.write(f"{w} ({score:.3f})")
    else:
        st.error("Word not found in vocabulary.")
