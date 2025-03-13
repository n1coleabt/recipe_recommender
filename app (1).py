import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the cleaned dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("allrecipes_cleaned.csv")  # Ensure this file is uploaded to GitHub
        df['ingredients'] = df['ingredients'].astype(str).str.lower()
        return df
    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")
        return pd.DataFrame(columns=['title', 'ingredients'])  # Return empty dataframe if error

df = load_data()

# Check if dataset is loaded
if df.empty:
    st.error("🚨 Dataset is empty! Please check the file `allrecipes_cleaned.csv`.")
else:
    # Convert ingredients into numerical features using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['ingredients'])

    # Function to recommend recipes based on input ingredients
    def recommend_recipes(user_ingredients, top_n=5):
        if not user_ingredients.strip():  # Handle empty input
            return pd.DataFrame(columns=['title', 'ingredients'])

        user_vec = vectorizer.transform([user_ingredients.lower()])
        sim_scores = cosine_similarity(user_vec, X).flatten()
        top_indices = sim_scores.argsort()[-top_n:][::-1]
        return df.iloc[top_indices][['title', 'ingredients']]

    # Streamlit UI
    st.title("🍽️ Recipe Recommender")
    st.write("🔍 Enter ingredients you have, and we'll suggest recipes!")

    # User input
    ingredients = st.text_input("📝 Enter ingredients (comma-separated):")

    if ingredients:
        recommended_recipes = recommend_recipes(ingredients)

        if recommended_recipes.empty:
            st.warning("⚠️ No matching recipes found. Try different ingredients!")
        else:
            st.success("✅ Recommended Recipes:")
            for _, row in recommended_recipes.iterrows():
                st.subheader(row['title'])
                st.write(f"**🛒 Ingredients:** {row['ingredients']}")
