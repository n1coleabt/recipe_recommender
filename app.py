import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load fine-tuned model and tokenizer
MODEL_PATH = "fine_tuned_recipe_model"  # Change this if hosting on Hugging Face
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

def generate_recipe(prompt):
    """Generates a recipe based on user input."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        output = model.generate(**inputs, max_length=512, num_return_sequences=1)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Streamlit UI
st.title("🍽️ AI Recipe Recommender")
st.markdown("Type a description of the dish you want, and the AI will generate a recipe!")

user_input = st.text_area("Describe your desired recipe:")
if st.button("Generate Recipe"):
    if user_input:
        with st.spinner("Cooking up something special..."):
            recipe = generate_recipe(user_input)
        st.subheader("Here's your AI-generated recipe:")
        st.write(recipe)
    else:
        st.warning("Please enter a description of your desired dish.")
