# Recipe Recommender System

Welcome to the **Recipe Recommender System**! This project is designed to help users discover and explore Japanese cuisine by recommending recipes based on user queries. The system scrapes recipe data from the web, processes it, and uses machine learning models to provide personalized recommendations.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Running the Project](#running-the-project)
4. [Running on Streamlit Cloud](#running-on-streamlit-cloud)
5. [Project Structure](#project-structure)
6. [Dependencies](#dependencies)
7. [Contributing](#contributing)
8. [License](#license)

---

## Project Overview

The Recipe Recommender System is a web-based application that allows users to search for Japanese recipes by entering an ingredient or dish name. The system uses web scraping to collect recipe data, processes it, and leverages machine learning models (fine-tuned GPT-2) and FAISS for efficient recipe retrieval. The user interface is built using Streamlit, making it easy to use and interact with.

### Key Features:
- **Web Scraping**: Collects recipe data from AllRecipes.
- **Data Processing**: Cleans and structures the data for machine learning.
- **Machine Learning**: Fine-tunes a GPT-2 model for recipe recommendations.
- **FAISS Integration**: Efficiently retrieves similar recipes based on user queries.
- **Streamlit UI**: A user-friendly interface for interacting with the system.

---

## Installation

To run this project, you'll need to set up your environment and install the required dependencies.

### Prerequisites:
- Python 3.8 or higher
- Git (optional, for cloning the repository)

### Steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/n1coleabt/recipe_recommender.git
   cd recipe_recommender
   ```

2. **Create a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Pre-trained Models**:
   - The project uses a fine-tuned GPT-2 model and FAISS index. Ensure these are downloaded and placed in the appropriate directories.
   - You can download the models from the Hugging Face Hub or generate them by running the provided scripts.

---

## Running the Project

Once the installation is complete, you can run the project using the following steps:

1. **Run the Streamlit App Locally**:
   ```bash
   streamlit run app.py
   ```

2. **Access the Web Interface**:
   - After running the above command, Streamlit will provide a local URL (usually `http://localhost:8501`).
   - Open the URL in your web browser to access the Recipe Recommender System.

3. **Enter a Query**:
   - In the web interface, enter an ingredient or dish name (e.g., "gyoza") to get recipe recommendations.
   - The system will display a list of recommended recipes along with their summaries, ingredients, and instructions.

---

## Running on Streamlit Cloud (RECOMMENDED)

You can also deploy this project on **Streamlit Cloud** for easy access and sharing. Here's how:

### Steps to Deploy on Streamlit Cloud:

1. **Create a Streamlit Cloud Account**:
   - If you don't have a Streamlit Cloud account, sign up at [Streamlit Cloud](https://streamlit.io/cloud).

2. **Connect Your GitHub Repository**:
   - Go to the Streamlit Cloud dashboard and click on "New App".
   - Connect your GitHub account and select the repository for this project (`recipe_recommender`).

3. **Configure the App**:
   - Set the branch to `main` (or the branch where your code is located).
   - Set the main file path to `app.py`.
   - Add any necessary environment variables (if required).

4. **Deploy the App**:
   - Click "Deploy" to deploy the app on Streamlit Cloud.
   - Once deployed, Streamlit Cloud will provide a public URL for your app.

5. **Access the App**:
   - Share the public URL with others to allow them to interact with the Recipe Recommender System.

---

## Project Structure

The project is organized as follows:

```
recipe_recommender/
│
├── app.py                  # Streamlit application for the recipe recommender
├── requirements.txt        # List of dependencies
├── JPNmaindishes_cleaned.csv  # Cleaned dataset of Japanese recipes
├── JPNrecipes.jsonl        # JSONL file containing recipe data
├── faiss_index.idx         # FAISS index for recipe retrieval
├── recipes_metadata.json   # Metadata for recipes (title, ingredients, etc.)
├── tokenized_recipes/      # Directory for tokenized recipe data
├── fine_tuned_recipe_model/ # Directory for the fine-tuned GPT-2 model
├── README.md               # This file
└── .gitignore              # Files and directories to ignore in Git
```

---

## Dependencies

The project relies on the following Python libraries:

- **Streamlit**: For building the web interface.
- **Transformers**: For fine-tuning and using the GPT-2 model.
- **FAISS**: For efficient similarity search.
- **Pandas**: For data manipulation and storage.
- **BeautifulSoup**: For web scraping.
- **Sentence Transformers**: For generating embeddings.
- **NumPy**: For numerical computations.

You can install all dependencies by running:
```bash
pip install -r requirements.txt
```

---

### Guidelines:
- Ensure your code follows PEP 8 style guidelines.
- Write clear and concise commit messages.
- Include tests for any new functionality.

---

## License

This project is licensed under the **MIT License**. Feel free to use, modify, and distribute the code as per the terms of the license.

---

## Acknowledgments

- **Hugging Face**: For providing the Transformers library and pre-trained models.
- **FAISS**: For efficient similarity search.
- **AllRecipes**: For the recipe data used in this project.

---

Enjoy exploring Japanese cuisine with the Recipe Recommender System! 🍜
