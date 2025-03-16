import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())

import streamlit as st
import faiss
import numpy as np
import torch
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

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
        with open("recipes_metadata.json", "r", encoding="utf-8") as f:
            recipes = json.load(f)
        return index, recipes
    except Exception as e:
        st.error(f"Error loading FAISS index: {e}")
        return None, None

# ✅ Load a smaller LLM model
@st.cache_resource()
def load_llm():
    model_name = "facebook/opt-1.3b"  # ✅ A much smaller model that fits in Streamlit Cloud memory
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True  # ✅ Prevents memory overload
    )
    return model, tokenizer

# Function to generate embeddings
def get_embedding(query):
    return embedding_model.encode([query], convert_to_numpy=True)

# Function to retrieve recipes
def retrieve_recipes(query, k=5):
    index, recipes = load_faiss_index()
    if index is None or recipes is None:
        return []
    
    query_embedding = get_embedding(query)
    
    # Ensure dimensions match
    if query_embedding.shape[1] != index.d:
        st.error(f"Dimension mismatch: FAISS expects {index.d}, but got {query_embedding.shape[1]}.")
        return []
    
    _, indices = index.search(query_embedding, k)
    
    results = [recipes[i] for i in indices[0] if i < len(recipes)]
    return results

# ✅ Function to generate LLM-based summaries
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
st.write("Enter an ingredient or dish to get **Japanese recipe** recommendations.")

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
st.markdown("Made with ❤️ using **Streamlit** & **FAISS** | Recipes sourced from AllRecipes")
