# app/streamlit_app.py

import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np
import pandas as pd
import pickle

# Load FAISS index, document embeddings, and documents
@st.cache_resource
def load_retriever():
    with open("data/embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)
    with open("data/docs.pkl", "rb") as f:
        docs = pickle.load(f)
    index = faiss.read_index("data/faiss_index.index")
    return embeddings, docs, index

# Load generator model
@st.cache_resource
def load_generator():
    return pipeline("text2text-generation", model="google/flan-t5-base")

# Retrieve top-k documents
def retrieve(query, index, docs, model, k=3):
    query_emb = model.encode([query])[0]
    D, I = index.search(np.array([query_emb]), k)
    return [docs[i] for i in I[0]]

# Main Streamlit app
def main():
    st.title("🧠 RAG Clinical Diagnosis Assistant")
    st.write("Ask a medical question to get AI-assisted diagnostic reasoning.")

    query = st.text_input("Enter clinical query (e.g., 'fever and chest pain'):")

    if query:
        st.info("Retrieving relevant documents...")
        emb_model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings, docs, index = load_retriever()
        generator = load_generator()
        retrieved_docs = retrieve(query, index, docs, emb_model)

        st.subheader("🔍 Retrieved Documents:")
        for i, doc in enumerate(retrieved_docs, 1):
            st.markdown(f"**Doc {i}:** {doc[:500]}...")

        st.subheader("🧠 Generated Diagnostic Summary:")
        prompt = f"Context: {' '.join(retrieved_docs)} \n\n Question: {query} \n\n Diagnosis:"
        response = generator(prompt, max_length=200, do_sample=True)[0]['generated_text']
        st.success(response)

if __name__ == "__main__":
    main()
