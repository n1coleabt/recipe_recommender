import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())

import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import os

# Set Streamlit page config
st.set_page_config(page_title="🍣 Japanese Recipe Recommender", layout="wide")

# Title and description
st.title("🍣 Japanese Recipe Recommender")
st.write("Discover delicious **Japanese main dishes** from [AllRecipes](https://www.allrecipes.com/recipes/17491/world-cuisine/asian/japanese/main-dishes/). Enter an ingredient or dish to get personalized recommendations!")

# Load the fine-tuned recipe model
model_name = "nabt1/fine_tuned_recipe_model"
try:
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="./model_cache")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./model_cache")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

if os.path.exists(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
else:
    st.error(f"Model directory `{model_name}` not found! Please upload the fine-tuned model.")
    st.stop()

# Load FAISS index and recipe texts
try:
    index = faiss.read_index("recipe_index.faiss")
    recipe_texts = np.load("recipe_texts.npy", allow_pickle=True)
except Exception as e:
    st.error(f"Error loading FAISS index or recipe texts: {e}")
    st.stop()

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# User input
st.markdown("### 🔎 Search for a Japanese Dish")
st.markdown("_Note: This recommender only provides recipes from the **[AllRecipes Japanese Main Dishes](https://www.allrecipes.com/recipes/17491/world-cuisine/asian/japanese/main-dishes/)** section._")

user_query = st.text_input("Enter an ingredient or dish:")

if user_query:
    # Convert input query to embedding
    query_embedding = embedding_model.encode([user_query], convert_to_numpy=True)

    # Search in FAISS
    k = 5  # Number of results
    D, I = index.search(query_embedding, k)

    # Display results
    if len(I[0]) > 0:
        st.subheader("🍜 Recommended Japanese Recipes:")
        for i in I[0]:
            st.write(f"- {recipe_texts[i]}")
    else:
        st.warning("No matching Japanese recipes found. Try a different ingredient or dish!")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using **Streamlit** & **FAISS** | Recipes sourced from [AllRecipes](https://www.allrecipes.com/recipes/17491/world-cuisine/asian/japanese/main-dishes/)")
# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit & FAISS")
