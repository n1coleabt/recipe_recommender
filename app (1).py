import streamlit as st
import faiss
import numpy as np
import json
from PyPDF2 import PdfWriter
from sentence_transformers import SentenceTransformer

# Load model and FAISS index
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("recipe_index.faiss")

# Load recipe data
with open("JPNrecipes.jsonl", "r", encoding="utf-8") as file:
    recipes = [json.loads(line) for line in file]

# Function to search recipes
def search_recipes(query, k=5):
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)
    return [recipes[i] for i in indices[0]]

# Function to create a PDF
def create_pdf(selected_recipes):
    pdf_writer = PdfWriter()
    for recipe in selected_recipes:
        pdf_writer.add_blank_page()
        pdf_writer.add_text(50, 750, f"Recipe: {recipe['name']}")
        pdf_writer.add_text(50, 700, f"Ingredients: {', '.join(recipe['ingredients'])}")
        pdf_writer.add_text(50, 650, f"Steps: {recipe['steps']}")
    
    with open("recommended_recipes.pdf", "wb") as pdf_file:
        pdf_writer.write(pdf_file)

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
      
