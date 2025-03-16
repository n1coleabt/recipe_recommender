import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import faiss
import torch
import json
import os

# Load the FAISS index and recipe metadata
@st.cache_resource()
def load_faiss_index():
    index = faiss.read_index("faiss_index.idx")
    with open("recipes_metadata.json", "r") as f:
        recipes = json.load(f)
    return index, recipes

# Load LLM for generating summaries
@st.cache_resource()
def load_llm():
    model_name = "mistralai/Mistral-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    return model, tokenizer

# Function to retrieve recipes
def retrieve_recipes(query, k=5):
    index, recipes = load_faiss_index()
    # Convert query to embedding (assuming you have a function for this)
    query_embedding = get_embedding(query)  # Implement this
    _, indices = index.search(query_embedding, k)
    results = [recipes[i] for i in indices[0]]
    return results

# Function to generate LLM-based summaries
def generate_summary(recipe):
    model, tokenizer = load_llm()
    prompt = f"""Summarize this Japanese recipe:
    {recipe['title']}\n\nIngredients: {', '.join(recipe['ingredients'])}\n\nInstructions: {recipe['instructions']}"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=150)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

# Streamlit UI
st.set_page_config(page_title="Recipe Recommender App", layout="centered")
st.title("🍜 Recipe Recommender (Japanese Cuisine Only)")
st.write("Enter an ingredient or dish to get **Japanese recipe** recommendations from [AllRecipes](https://www.allrecipes.com/recipes/17491/world-cuisine/asian/japanese/main-dishes/).")

query = st.text_input("Enter an ingredient or dish:")
if query:
    recipes = retrieve_recipes(query)
    for recipe in recipes:
        summary = generate_summary(recipe)
        st.subheader(recipe['title'])
        st.write("**Summary:**", summary)
        st.write("**Ingredients:**", ", ".join(recipe['ingredients']))
        st.write("[View Full Recipe]({})".format(recipe['url']))

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using **Streamlit** & **FAISS** | Recipes sourced from [AllRecipes](https://www.allrecipes.com/recipes/17491/world-cuisine/asian/japanese/main-dishes/)")
