import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())

import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from recipe_recommender import search_recipes

# Streamlit UI
st.title("🍽️ Recipe Recommender App")
st.write("Find the best recipes based on your ingredients or preferences.")

# User input
user_query = st.text_input("Enter an ingredient or dish name:")
if st.button("Search"):
    if user_query:
        results = search_recipes(user_query)
        if results:
            st.write("### Recommended Recipes:")
            selected_recipes = []
            for recipe in results:
                st.write(f"**{recipe['name']}**")
                st.write(f"Ingredients: {', '.join(recipe['ingredients'])}")
                st.write(f"Steps: {recipe['steps'][:200]}...")  # Show only first 200 chars
                if st.checkbox(f"Select {recipe['name']} for PDF"):
                    selected_recipes.append(recipe)

            if selected_recipes:
                if st.button("Download Selected Recipes as PDF"):
                    create_pdf(selected_recipes)
                    with open("recommended_recipes.pdf", "rb") as pdf_file:
                        st.download_button("Download PDF", pdf_file, "recipes.pdf")
        else:
            st.write("No matching recipes found.")
    else:
        st.write("Please enter a search query.")
