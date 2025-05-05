import streamlit as st
import requests

st.set_page_config(
    page_title="NLP preprocessing demo",
    layout = "wide",)

st.title("NLP Preprocessing Demo")

text = st.text_area("Enter text here", height=200)
operation = st.selectbox("Select operation", ["tokenize", "lemmatize", "pos_tag", "ner"])

if st.button("Process"):
    if text:
        payload = {
            "text": text,
            "operation": operation
        }
        try:
            response = requests.post("http://localhost:8000/process_text", json=payload)
            if response.status_code == 200:
                st.subheader("Result:")
                st.json(response.json()["result"])  
            else:
                st.error("Error processing text")
        except requests.exceptions.RequestException as e:
            st.error(f"Error processing text: {e}")
    else:
        st.warning("Please enter some text to process.")
