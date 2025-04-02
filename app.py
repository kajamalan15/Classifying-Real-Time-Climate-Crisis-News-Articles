import streamlit as st
import joblib
import spacy

# Load spaCy model for text preprocessing
nlp = spacy.load("en_core_web_sm")

# Function to preprocess the text
def preprocess_text(text):
    doc = nlp(text.lower())  # Lowercase and tokenize
    words = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]  # Remove stopwords and lemmatize
    return " ".join(words)

# Load the model and vectorizer correctly
loaded_model = joblib.load(r"C:\Users\Kajamalan\Downloads\my_climate_news_model (3).pkl")
loaded_vectorizer = joblib.load(r"C:\Users\Kajamalan\Downloads\my_climate_news_vectorizer (2).pkl")

# Function to predict directly
def predict_category(text):
    processed_text = preprocess_text(text)
    vectorized_text = loaded_vectorizer.transform([processed_text])
    prediction = loaded_model.predict(vectorized_text)[0]  # No mapping needed if model returns label
    return prediction

# Streamlit UI
st.set_page_config(page_title="Climate Crisis News Classifier", layout="centered")

st.markdown("""
      <style>
        /* Fix the app width */
        .block-container {
            max-width: 700px;
            margin: auto;
            padding-top: 30px;
        }
        .result {
            font-size: 18px;
            font-weight: bold;
            text-align: center;
            padding: 8px;
            border-radius: 8px;
            margin-top: 15px;
            background-color: lightblue;
        }
        .small-text {
            font-size: 13px;
            color: gray;
            text-align: right;
        }
        .stTextArea textarea {
            border-radius: 8px;
            padding: 10px;
            border: 1px solid #ccc;
        }
        .stButton > button {
            width: 100%;
            border-radius: 8px;
            padding: 10px;
            background-color: #27ae60;
            color: white;
            font-weight: bold;
        }
        .stButton > button:hover {
            background-color: #1e8449;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🌍 Climate Crisis News Classifier")

text_input = st.text_area("Enter your climate news article:", placeholder="e.g., 'New research shows rising sea levels are a major concern.'")

st.markdown(f"<p class='small-text'>Character count: {len(text_input)}</p>", unsafe_allow_html=True)

if st.button("Classify Article"):
    if text_input.strip():
        category = predict_category(text_input)
        st.markdown(f'<div class="result" style="background-color: lightblue; color:black;">Predicted Category: {category}</div>', unsafe_allow_html=True)
    else:
        st.warning("Please enter an article to classify.")
