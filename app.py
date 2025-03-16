import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())

import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import faiss
import numpy as np
import torch
import json
import os

# Check if FAISS index exists
if not os.path.exists("faiss_index.idx"):
    # Dummy data for vectorized recipes (replace with actual embeddings)
    recipe_embeddings = np.random.rand(100, 768).astype("float32")

    # Create a FAISS index
    index = faiss.IndexFlatL2(recipe_embeddings.shape[1])
    index.add(recipe_embeddings)

    # Save the index
    faiss.write_index(index, "faiss_index.idx")
    print("FAISS index created and saved.")

# Load FAISS index and recipe metadata
@st.cache_resource()
def load_faiss_index():
    try:
        index = faiss.read_index("faiss_index.idx")
        with open("recipes_metadata.json", "r") as f:
            recipes = json.load(f)
        return index, recipes
    except Exception as e:
        st.error(f"Error loading FAISS index: {e}")
        return None, None

# Load LLM for generating summaries
@st.cache_resource()
def load_llm():
    model_name = "mistralai/Mistral-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    return model, tokenizer

# Function to generate embeddings for query (Placeholder: replace with real embedding model)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(query):
    return embedding_model.encode([query], convert_to_numpy=True)  # This returns (1, 384)

# Function to retrieve recipes
def retrieve_recipes(query, k=5):
    index, recipes = load_faiss_index()
    if index is None or recipes is None:
        return []
    
    query_embedding = get_embedding(query)  # Convert query to embedding
    _, indices = index.search(query_embedding, k)
    
    results = [recipes[i] for i in indices[0] if i < len(recipes)]
    return results

# Function to generate LLM-based summaries
def generate_summary(recipe):
    model, tokenizer = load_llm()
    prompt = (
    f"Summarize this Japanese recipe: {recipe['title']}\n\n"
    f"Ingredients: {', '.join(recipe['ingredients'])}\n\n"
    f"Instructions: {recipe['instructions']}"
)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=150)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Streamlit UI
st.set_page_config(page_title="Recipe Recommender App", layout="centered")
st.title("🍜 Recipe Recommender (Japanese Cuisine Only)")
st.write("Enter an ingredient or dish to get **Japanese recipe** recommendations from [AllRecipes](https://www.allrecipes.com/recipes/17491/world-cuisine/asian/japanese/main-dishes/).")

query = st.text_input("Enter an ingredient or dish:")
if query:
    recipes = retrieve_recipes(query)
    
    if recipes:
        for recipe in recipes:
            summary = generate_summary(recipe)
            st.subheader(recipe['title'])
            st.write("**Summary:**", summary)
            st.write("**Ingredients:**", ", ".join(recipe['ingredients']))
            st.write(f"[View Full Recipe]({recipe['url']})")
    else:
        st.warning("No recipes found. Try a different ingredient or dish.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using **Streamlit** & **FAISS** | Recipes sourced from [AllRecipes](https://www.allrecipes.com/recipes/17491/world-cuisine/asian/japanese/main-dishes/)")
