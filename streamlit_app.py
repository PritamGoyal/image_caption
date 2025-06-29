import streamlit as st
import numpy as np
from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.preprocessing.sequence import pad_sequences

from pickle import load
import io

# Load the tokenizer and trained model
import json
from collections import OrderedDict
from tensorflow.keras.preprocessing.text import Tokenizer

def load_tokenizer_from_json(path):
    with open(path, 'r') as f:
        tokenizer_data = json.load(f)

    tokenizer = Tokenizer()
    tokenizer.word_counts = OrderedDict(tokenizer_data["word_counts"])
    tokenizer.word_docs = tokenizer_data["word_docs"]
    tokenizer.word_index = tokenizer_data["word_index"]
    tokenizer.index_word = {int(k): v for k, v in tokenizer_data["index_word"].items()}
    tokenizer.num_words = tokenizer_data.get("num_words")
    return tokenizer



tokenizer = load_tokenizer_from_json("tokenizer.json")

    
model = load_model("model_1.h5")
xception_model = Xception(include_top=False, pooling="avg")

# Image preprocessing and feature extraction
def extract_features_1(img, model):
    image = img.resize((299, 299))
    image = np.array(image)
    if image.shape[2] == 4:  # Convert RGBA to RGB
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)
    image = image / 127.5 - 1.0
    feature = model.predict(image)
    return feature

# Word mapping
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Caption generation
def generate_desc(model, tokenizer, photo, max_length=32):
    in_text = 'start'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo, sequence], verbose=0)
        pred_id = np.argmax(pred)
        word = word_for_id(pred_id, tokenizer)
        if word is None or word == 'end':
            break
        in_text += ' ' + word
    return in_text.replace('start', '').replace('end', '').strip()

# Streamlit UI
st.title("🖼️ Image Caption Generator")
st.write("Upload an image to get a caption generated using your trained model.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Generate Caption"):
        with st.spinner("Generating..."):
            features = extract_features_1(image, xception_model)
            caption = generate_desc(model, tokenizer, features)
            st.success(f"📝 Caption: {caption}")
