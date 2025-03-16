import asyncio
import os
import streamlit as st
import torch
import faiss
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

# Ensure proper event loop handling
asyncio.set_event_loop(asyncio.new_event_loop())

# Set Streamlit page config
st.set_page_config(page_title="Recipe Recommender", layout="wide")

# Title and description
st.title("🍽️ Recipe Recommender App")
st.write("Enter an ingredient or dish to get recipe recommendations!")

# Model name from Hugging Face
MODEL_NAME = "nabt1/fine_tuned_recipe_model"

# Load the fine-tuned recipe model
try:
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir="./model_cache")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir="./model_cache")
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Check if FAISS index and recipe texts exist
INDEX_PATH = "recipe_index.faiss"
TEXTS_PATH = "recipe_texts.npy"

if not (os.path.exists(INDEX_PATH) and os.path.exists(TEXTS_PATH)):
    st.error("❌ FAISS index or recipe texts not found! Ensure the required files are available.")
    st.stop()

# Load FAISS index and recipe texts
try:
    index = faiss.read_index(INDEX_PATH)
    recipe_texts = np.load(TEXTS_PATH, allow_pickle=True)
    st.success("✅ FAISS index and recipe texts loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading FAISS index or recipe texts: {e}")
    st.stop()

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# User input
user_query = st.text_input("🔎 Enter an ingredient or dish:")

if user_query:
    # Convert input query to embedding
    query_embedding = embedding_model.encode([user_query], convert_to_numpy=True)

    # Search in FAISS
    k = 5  # Number of results
    D, I = index.search(query_embedding, k)

    # Display results
    if len(I[0]) > 0:
        st.subheader("🥘 Recommended Recipes:")
        for i in I[0]:
            st.write(f"- {recipe_texts[i]}")
    else:
        st.warning("⚠️ No matching recipes found. Try a different ingredient or dish!")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit & FAISS")
