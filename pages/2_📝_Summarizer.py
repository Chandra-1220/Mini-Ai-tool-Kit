import streamlit as st
from utils.models import load_summarizer

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Text Summarizer",
    page_icon="📝",
    layout="wide"
)

summarizer = load_summarizer()

# -----------------------------
# Header
# -----------------------------
st.title("📝 AI Text Summarizer")
st.write("Convert long articles into short, meaningful summaries.")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Settings")

max_length = st.sidebar.slider(
    "Maximum Summary Length",
    30,
    150,
    80
)

min_length = st.sidebar.slider(
    "Minimum Summary Length",
    10,
    80,
    30
)

# -----------------------------
# Example
# -----------------------------
example_text = """
Artificial Intelligence (AI) is transforming industries by enabling machines to perform tasks that normally require human intelligence. AI systems can analyze large amounts of data, recognize patterns, make decisions, and continuously improve through learning. Applications of AI include healthcare, finance, education, transportation, customer service, and robotics. As AI technology advances, it is becoming an essential part of modern businesses and everyday life.
"""

if "summary_text" not in st.session_state:
    st.session_state.summary_text = st.session_state.get("input_text", "")
# -----------------------------
# Input
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    if st.button("🔄 Use Home Input", use_container_width=True):
        st.session_state.summary_text = st.session_state.get("input_text", "")

with col2:
    if st.button("📋 Load Example", use_container_width=True):
        st.session_state.summary_text = example_text
article = st.text_area(
    "Enter your article",
    key="summary_text",
    height=300,
    placeholder="Paste your article here..."
)

# -----------------------------
# Summarize
# -----------------------------
if st.button("✨ Generate Summary", use_container_width=True):

    if article.strip() == "":
        st.warning("Please enter some text.")

    elif len(article.split()) < 30:
        st.warning("Please enter at least 30 words.")

    else:

        with st.spinner("Generating summary..."):

            result = summarizer(
                article,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )

            summary = result[0]["summary_text"]

        st.success("Summary Generated Successfully!")

        st.subheader("Summary")

        st.success(summary)

        # -----------------------------
        # Metrics
        # -----------------------------

        input_words = len(article.split())
        output_words = len(summary.split())

        reduction = round(
            (1 - output_words / input_words) * 100,
            1
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Input Words",
            input_words
        )

        col2.metric(
            "Summary Words",
            output_words
        )

        col3.metric(
            "Reduction",
            f"{reduction}%"
        )

        st.download_button(
            "📥 Download Summary",
            summary,
            file_name="summary.txt",
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
    if st.button("💬 Question Answering ➜", use_container_width=True):
        st.switch_page("pages/3_💬_Question_Answering.py")
