import streamlit as st
import joblib
import re
import string
import os

# --- 1. Define File Paths ---

MODELS_FOLDER = 'models'
VECTORIZER_PATH = os.path.join(MODELS_FOLDER, 'vectorizer.joblib')
MODEL_PATH = os.path.join(MODELS_FOLDER, 'model.joblib')

# --- 2. Text Cleaning Function ---

def clean_text(text):
    """
    1. Converts to lowercase
    2. Removes punctuation
    3. Strips extra whitespace
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = " ".join(text.split())
    return text

# --- 3. Load Model and Vectorizer (with Caching) ---

@st.cache_resource
def load_model_and_vectorizer():
    """
    Loads the trained Logistic Regression model and TF-IDF vectorizer.
    """
    print("Loading model and vectorizer...")
    try:
        vectorizer = joblib.load(VECTORIZER_PATH)
        model = joblib.load(MODEL_PATH)
        print("Model and vectorizer loaded successfully.")
        return model, vectorizer
    except FileNotFoundError:
        st.error(f"Error: Model or vectorizer files not found.")
        st.error(f"Searched paths: {MODEL_PATH}, {VECTORIZER_PATH}")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading models: {e}")
        st.stop()


model, vectorizer = load_model_and_vectorizer()

# --- 4. Streamlit App Interface ---

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

# Title
st.title("📰 Interactive Fake News Detector")
st.markdown("Enter a news article's text below to classify it as Real or Fake.")

# --- 5. User Input and Prediction ---
st.header("Enter News Text")
news_text = st.text_area("Paste the full text of the news article here:", height=300, 
                         placeholder="Example: 'Scientists today announced the discovery of a new planet...'")

# Prediction button
if st.button("Classify News", type="primary", use_container_width=True):
    if not news_text.strip():
        # If text area is empty or just whitespace
        st.warning("Please enter some news text to classify.")
    else:
        # --- 6. Prediction Logic ---
        # 1. Clean the user's input text
        cleaned_input = clean_text(news_text)
        
        # 2. Transform the text using the loaded vectorizer
       
        vectorized_input = vectorizer.transform([cleaned_input])
        
        # 3. Make a prediction
        prediction = model.predict(vectorized_input)
        
        # 4. Get the probability (confidence)

        probability = model.predict_proba(vectorized_input)
        
        confidence_fake = probability[0][0]
        confidence_real = probability[0][1]

        # --- 7. Display the Result ---
        st.header("Classification Result")
        
        if prediction[0] == 0: # Fake
            st.error(f"**This article is classified as FAKE.**", icon="❌")
            st.markdown(f"**Confidence:** {confidence_fake:.2%}")
        else: # Real
            st.success(f"**This article is classified as REAL.**", icon="✅")
            st.markdown(f"**Confidence:** {confidence_real:.2%}")

# --- 8. Sidebar / Footer ---
st.sidebar.header("About This Project")
st.sidebar.markdown("""
This app uses a **Logistic Regression** model trained on the 
[Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset).

**How it works:**
1.  **Input:** You provide the text of a news article.
2.  **Cleaning:** The text is cleaned (punctuation removed, all lowercase).
3.  **Vectorization:** The text is converted into numbers using a TF-IDF Vectorizer.
4.  **Prediction:** A pre-trained model classifies the text as 'Real' or 'Fake'.
""")
st.sidebar.markdown("---")
st.sidebar.markdown("Made by Ashtam Pati Tiwari")
