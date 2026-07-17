import streamlit as st
from utils.models import load_translation_model

# ----------------------------------
# Page Configuration
# ----------------------------------
st.set_page_config(
    page_title="Language Translator",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 AI Language Translator")
st.write(
    "Translate text between multiple languages using Hugging Face Transformers."
)

# ----------------------------------
# Language Pairs
# ----------------------------------
LANGUAGE_PAIRS = {
    "English → French": "Helsinki-NLP/opus-mt-en-fr",
    "English → German": "Helsinki-NLP/opus-mt-en-de",
    "English → Spanish": "Helsinki-NLP/opus-mt-en-es",
    "English → Hindi": "Helsinki-NLP/opus-mt-en-hi",
    "French → English": "Helsinki-NLP/opus-mt-fr-en",
    "German → English": "Helsinki-NLP/opus-mt-de-en",
    "Spanish → English": "Helsinki-NLP/opus-mt-es-en",
    "Hindi → English": "Helsinki-NLP/opus-mt-hi-en",
}

# ----------------------------------
# Sidebar
# ----------------------------------
st.sidebar.header("Translation Settings")

translation_type = st.sidebar.selectbox(
    "Translation Direction",
    list(LANGUAGE_PAIRS.keys())
)

# ----------------------------------
# Example
# ----------------------------------
example = "Artificial Intelligence is transforming the world by enabling machines to learn from data."

if st.button("📋 Load Example"):
    st.session_state.translation_text = example

# ----------------------------------
# Input Text
# ----------------------------------
text = st.text_area(
    "Enter Text",
    value=st.session_state.get("translation_text", ""),
    height=220,
    placeholder="Type or paste your text here..."
)

# ----------------------------------
# Translate
# ----------------------------------
if st.button("🌍 Translate", use_container_width=True):

    if text.strip() == "":
        st.warning("Please enter some text.")

    else:

        with st.spinner("Translating..."):

            translator = load_translation_model(
                LANGUAGE_PAIRS[translation_type]
            )

            result = translator(text)

            translated_text = result[0]["translation_text"]

        st.success("Translation Completed!")

        st.subheader("Translated Text")

        st.success(translated_text)

        # -----------------------------
        # Metrics
        # -----------------------------
        input_words = len(text.split())
        output_words = len(translated_text.split())

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Input Words",
                input_words
            )

        with col2:
            st.metric(
                "Output Words",
                output_words
            )

        # -----------------------------
        # Original Text
        # -----------------------------
        st.subheader("Original Text")

        st.info(text)

        # -----------------------------
        # Download
        # -----------------------------
        st.download_button(
            label="📥 Download Translation",
            data=translated_text,
            file_name="translation.txt",
            mime="text/plain"
        )

# ----------------------------------
# Navigation
# ----------------------------------
st.divider()

col1, col2 = st.columns(2)

with col1:
    if st.button("🏠 Home", use_container_width=True):
        st.switch_page("Home.py")

with col2:
    if st.button("🏷️ Named Entity Recognition ➜", use_container_width=True):
        st.switch_page("pages/5_🏷️_NER.py")
