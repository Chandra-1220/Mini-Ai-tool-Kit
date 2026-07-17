import streamlit as st
import pandas as pd
from utils.models import load_zero_shot

# ----------------------------------
# Page Configuration
# ----------------------------------
st.set_page_config(
    page_title="Zero-Shot Classification",
    page_icon="🎯",
    layout="wide"
)

classifier = load_zero_shot()

# ----------------------------------
# Header
# ----------------------------------
st.title("🎯 Zero-Shot Classification")

st.write(
    "Classify text into custom categories without training a model."
)

# ----------------------------------
# Sidebar
# ----------------------------------
st.sidebar.header("Instructions")

st.sidebar.info("""
1. Enter your text.
2. Enter labels separated by commas.
3. Click Classify.
""")

# ----------------------------------
# Example
# ----------------------------------
example_text = """
Artificial Intelligence is transforming healthcare by helping doctors detect diseases faster and more accurately.
"""

example_labels = "Technology, Healthcare, Sports, Finance, Education"

# Initialize from Home Input
if "zero_text" not in st.session_state:
    st.session_state.zero_text = st.session_state.get("input_text", "")

if "zero_labels" not in st.session_state:
    st.session_state.zero_labels = ""

# ----------------------------------
# Buttons
# ----------------------------------
col1, col2 = st.columns(2)

with col1:
    if st.button("🔄 Use Home Input", use_container_width=True):
        st.session_state.zero_text = st.session_state.get("input_text", "")

with col2:
    if st.button("📋 Load Example", use_container_width=True):
        st.session_state.zero_text = example_text
        st.session_state.zero_labels = example_labels

# ----------------------------------
# Input
# ----------------------------------
text = st.text_area(
    "Enter Text",
    key="zero_text",
    height=220,
    placeholder="Type or paste your text..."
)

labels = st.text_input(
    "Candidate Labels (comma separated)",
    key="zero_labels",
    placeholder="Technology, Sports, Finance"
)
# ----------------------------------
# Classification
# ----------------------------------
if st.button("🚀 Classify", use_container_width=True):

    if text.strip() == "":
        st.warning("Please enter some text.")

    elif labels.strip() == "":
        st.warning("Please enter at least one label.")

    else:

        candidate_labels = [
            label.strip()
            for label in labels.split(",")
            if label.strip()
        ]

        with st.spinner("Classifying..."):

            result = classifier(
                text,
                candidate_labels
            )

        st.success("Classification Completed!")

        df = pd.DataFrame({
            "Label": result["labels"],
            "Confidence": [
                f"{score:.2%}"
                for score in result["scores"]
            ]
        })

        st.subheader("Prediction Results")

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Top Prediction")

        st.success(
            f"🏆 {result['labels'][0]} ({result['scores'][0]:.2%})"
        )

        st.metric(
            "Labels Evaluated",
            len(candidate_labels)
        )

        st.download_button(
            "📥 Download Results",
            df.to_csv(index=False),
            file_name="zero_shot_results.csv",
            mime="text/csv"
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
    if st.button("😊 Sentiment Analysis ➜", use_container_width=True):
        st.switch_page("pages/1_😊_Sentiment.py")
