import streamlit as st
import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_rnn_model():
    return load_model("model/sentiment_rnn.keras")

model = load_rnn_model()

# -------------------------------
# Load Tokenizer
# -------------------------------
with open("model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# -------------------------------
# Load Max Length
# -------------------------------
with open("model/maxlen.pkl", "rb") as f:
    MAX_LENGTH = pickle.load(f)

# -------------------------------
# Title
# -------------------------------
st.title("🎬 Movie Review Sentiment Analysis")
st.write("Predict whether a movie review is **Positive** or **Negative** using a Simple RNN model.")

st.divider()

# -------------------------------
# User Input
# -------------------------------
review = st.text_area(
    "Enter a Movie Review",
    height=180,
    placeholder="Example: This movie was amazing. The acting and story were excellent!"
)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a movie review.")
    else:

        sequence = tokenizer.texts_to_sequences([review])

        padded = pad_sequences(
            sequence,
            maxlen=MAX_LENGTH,
            padding="post",
            truncating="post"
        )

        prediction = model.predict(padded, verbose=0)[0][0]

        if prediction >= 0.5:
            sentiment = "Positive 😊"
            confidence = prediction * 100
            st.success(f"Prediction: {sentiment}")
        else:
            sentiment = "Negative 😞"
            confidence = (1 - prediction) * 100
            st.error(f"Prediction: {sentiment}")

        st.info(f"Confidence: {confidence:.2f}%")