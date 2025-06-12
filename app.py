from keras import backend as K
K.clear_session()
import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from keras.preprocessing.sequence import pad_sequences



# --- Load Model and Tokenizer ---
model = tf.keras.models.load_model("lstm_fake_news_detection_model.h5")
# No need to compile the model for inference
with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

max_len = 250  # same as during training

# --- Streamlit App ---
st.set_page_config(page_title="📰 Fake News Detector", layout="centered")

st.title("📰 Fake News Detection with LSTM")
st.markdown("Enter a news article below and the model will predict whether it's **real or fake**.")

# Input box
user_input = st.text_area("News Article Text", height=200)

# Predict button
if st.button("Detect"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Preprocess input
        seq = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')

        # Make prediction
        prediction = model.predict(padded)[0][0] * 10

        # Label prediction
        label = "🟥 Fake News" if prediction < 0.5 else "🟩 Real News"

        # Confidence calculation
        confidence = round(float(prediction), 4) if prediction > 0.5 else round(1 - float(prediction), 4)

        # Display result
        st.write(f"**{label}**")


