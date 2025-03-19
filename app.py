import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())

import streamlit as st
import faiss
import pandas as pd
import numpy as np
import torch
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Recipe Recommender App", layout="centered")

# Load Sentence Transformer Model (For FAISS Search)
@st.cache_resource()
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

# Load FAISS index and recipes metadata
@st.cache_resource()
def load_faiss_index():
    try:
        index = faiss.read_index("faiss_index.idx")

        # Ensure 'recipes_metadata.json' exists
        json_file = "recipes_metadata.json"
        if not os.path.exists(json_file):
            st.warning(f"{json_file} not found. Generating from CSV...")

            # Load CSV and check data
            csv_file = "JPNmaindishes_cleaned.csv"
            if not os.path.exists(csv_file):
                st.error(f"Error: {csv_file} not found. Please upload it.")
                return None, None

            df = pd.read_csv(csv_file)

            # Check if required columns exist
            required_columns = {"name", "summary", "ingredients", "process", "url"}
            if not required_columns.issubset(df.columns):
                st.error(f"Error: CSV must contain columns: {', '.join(required_columns)}")
                return None, None

            # Convert DataFrame to JSON
            recipes = df.to_dict(orient="records")
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(recipes, f, indent=4)

        # Load recipes JSON
        with open(json_file, "r", encoding="utf-8") as f:
            recipes = json.load(f)

        return index, recipes

    except Exception as e:
        st.error(f"Error loading FAISS index: {e}")
        return None, None

# Load a smaller LLM model
@st.cache_resource()
def load_llm():
    model_name = "facebook/opt-350m"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Enables lower precision for reduced memory usage
        device_map="auto",
        offload_folder="./offload"
    )

    return model, tokenizer


# Load FAISS and LLM globally to avoid repeated loading
index, recipes = load_faiss_index()
model, tokenizer = load_llm()

# Function to generate embeddings
def get_embedding(query):
    return embedding_model.encode([query], convert_to_numpy=True).reshape(1, -1)

# Function to retrieve recipes
def retrieve_recipes(query, k=5):
    if index is None or recipes is None:
        return []

    query_embedding = get_embedding(query)
    _, indices = index.search(query_embedding, k)

    # Retrieve matching recipes
    results = [recipes[i] for i in indices[0] if i < len(recipes)]
    
    return results

# Function to clean up text fields
def clean_text(value, default="N/A"):
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default

# Function to generate structured summary
def generate_summary(recipe):
    title = clean_text(recipe.get("name"))
    summary = clean_text(recipe.get("summary"))
    ingredients_raw = clean_text(recipe.get("ingredients"))
    instructions_raw = clean_text(recipe.get("process"))

    # Convert ingredients to a formatted list
    if isinstance(ingredients_raw, str):
        ingredients_list = ingredients_raw.split(" | ")
    else:
        ingredients_list = ["N/A"]

    formatted_ingredients = "\n".join([f"- {ing.strip()}" for ing in ingredients_list if ing.strip()])

    # Convert instructions to a formatted step-by-step list
    if isinstance(instructions_raw, str):
        instructions_list = instructions_raw.split(" | ")
    else:
        instructions_list = ["N/A"]

    formatted_instructions = "\n".join([f"{i+1}. {step.strip()}" for i, step in enumerate(instructions_list) if step.strip()])

    return title, summary, formatted_ingredients, formatted_instructions

# Streamlit UI
st.title("🍜 Recipe Recommender (Japanese Cuisine Only)")
st.write("Enter an ingredient or dish to get **Japanese recipe** recommendations.")

query = st.text_input("Enter an ingredient or dish:")
if query:
    recipes = retrieve_recipes(query)

    if recipes:
        for recipe in recipes:
            title, summary, formatted_ingredients, formatted_instructions = generate_summary(recipe)
            
            st.subheader(title)
            st.write("**Summary:**", summary)
            st.write("**Ingredients:**")
            st.markdown(formatted_ingredients)
            st.write("**Instructions:**")
            st.markdown(formatted_instructions)
            st.write(f"[View Full Recipe]({recipe.get('url', '#')})")
    else:
        st.warning("No recipes found. Try a different ingredient or dish.")

# Footer
st.markdown("---")
st.markdown("Recipes sourced from AllRecipes")
