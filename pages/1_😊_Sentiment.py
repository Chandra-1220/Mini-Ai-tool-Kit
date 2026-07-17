import streamlit as st
from utils.models import load_sentiment_model

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="😊",
    layout="wide"
)

classifier = load_sentiment_model()

# -----------------------------
# Header
# -----------------------------
st.title("😊 Sentiment Analysis")
st.write("Analyze the sentiment of any text using Hugging Face Transformers.")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("About")
st.sidebar.info(
    """
This model predicts whether a piece of text expresses a:

- 😊 Positive sentiment
- 😞 Negative sentiment
"""
)

# -----------------------------
# Example Button
# -----------------------------
example_text = (
    "I absolutely love Streamlit! "
    "It makes building AI applications simple and enjoyable."
)

# Always load latest Home input
st.session_state.sentiment_text = st.session_state.get("input_text", "")

# Override only if example is requested
if st.button("📋 Load Example"):
    st.session_state.sentiment_text = example_text

text = st.text_area(
    "Enter your text",
    key="sentiment_text",
    height=180
)

# Keep session state updated if user edits the text
st.session_state.page_text = text

# -----------------------------
# Analyze Button
# -----------------------------
if st.button("🔍 Analyze Sentiment", use_container_width=True):

    if text.strip() == "":
        st.warning("Please enter some text.")
    else:

        with st.spinner("Analyzing..."):

            result = classifier(text)[0]

            label = result["label"]
            score = result["score"]

        st.success("Analysis Completed!")

        st.subheader("Prediction")

        if label == "POSITIVE":
            st.success(f"😊 Positive ({score:.2%})")
        else:
            st.error(f"😞 Negative ({score:.2%})")

        st.progress(float(score))

        st.metric(
            "Confidence Score",
            f"{score:.2%}"
        )

        st.subheader("Input Text")

        st.info(text)

        st.download_button(
            label="📥 Download Result",
            data=f"Sentiment: {label}\nConfidence: {score:.2%}\n\n{text}",
            file_name="sentiment_result.txt",
            mime="text/plain"
        )

# -----------------------------
# Navigation
# -----------------------------
st.divider()

col1, col2 = st.columns(2)

with col1:
    if st.button("🏠 Home", use_container_width=True):
        st.switch_page("Home.py")

with col2:
    if st.button("📝 Text Summarizer ➜", use_container_width=True):
        st.switch_page("pages/2_📝_Summarizer.py")
