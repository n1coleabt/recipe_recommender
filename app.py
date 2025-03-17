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
            required_columns = {"title", "ingredients", "instructions", "url"}
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
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
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

    results = [recipes[i] for i in indices[0] if i < len(recipes)]
    return results

# Function to generate LLM-based summaries
def generate_summary(recipe):
    title = recipe.get("title", "Unknown Recipe")  # Ensure title exists
    ingredients = recipe.get("ingredients", [])
    instructions = recipe.get("instructions", "No instructions available.")

    # Ensure ingredients & instructions are strings
    ingredients_str = ", ".join(ingredients) if isinstance(ingredients, list) else "No ingredients available."
    instructions_str = instructions if isinstance(instructions, str) else "No instructions available."

    # Truncate ingredients and instructions
    max_ingredients_length = 150  # Reduce length further
    max_instructions_length = 250

    if len(ingredients_str) > max_ingredients_length:
        ingredients_str = ingredients_str[:max_ingredients_length] + "..."
    
    if len(instructions_str) > max_instructions_length:
        instructions_str = instructions_str[:max_instructions_length] + "..."

    # Create prompt with truncated content
    prompt = (
        f"Summarize this Japanese recipe: {title}\n\n"
        f"Ingredients: {ingredients_str}\n\n"
        f"Instructions: {instructions_str}"
    )

    # Tokenize and limit input size
    model, tokenizer = load_llm()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(model.device)  # Further reduced max_length

    # Generate text with adjusted parameters
    outputs = model.generate(
        **inputs,
        max_length=50,  # Reduce max length
        min_length=20,  # Ensure reasonable output length
        do_sample=True,
        temperature=0.7
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Streamlit UI
st.title("🍜 Recipe Recommender (Japanese Cuisine Only)")
st.write("Enter an ingredient or dish to get **Japanese recipe** recommendations.")

query = st.text_input("Enter an ingredient or dish:")
if query:
    recipes = retrieve_recipes(query)

    if recipes:
        for recipe in recipes:
            summary = generate_summary(recipe)
            st.subheader(recipe.get("title", "Unknown Recipe"))  # Safely get title
            st.write("**Summary:**", summary)
            st.write("**Ingredients:**", ", ".join(recipe.get("ingredients", ["No ingredients available."])))
            st.write(f"[View Full Recipe]({recipe.get('url', '#')})")
    else:
        st.warning("No recipes found. Try a different ingredient or dish.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using **Streamlit** & **FAISS** | Recipes sourced from AllRecipes")
