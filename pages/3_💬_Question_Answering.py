import streamlit as st
from utils.models import load_qa_model

# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(
    page_title="Question Answering",
    page_icon="💬",
    layout="wide"
)

qa_pipeline = load_qa_model()

# -----------------------------------
# Header
# -----------------------------------
st.title("💬 AI Question Answering")
st.write(
    "Provide a context paragraph and ask a question. "
    "The AI will extract the answer from the context."
)

# -----------------------------------
# Example
# -----------------------------------
example_context = """
Artificial Intelligence (AI) is a branch of computer science that enables
machines to perform tasks that normally require human intelligence. AI is
widely used in healthcare, finance, education, robotics, and autonomous
vehicles. Machine Learning is a subset of AI that allows computers to learn
from data without being explicitly programmed.
"""

example_question = "What is Machine Learning?"

# Initialize from Home Input
if "qa_context" not in st.session_state:
    st.session_state.qa_context = st.session_state.get("input_text", "")

if "qa_question" not in st.session_state:
    st.session_state.qa_question = ""

# -----------------------------------
# Buttons
# -----------------------------------
col1, col2 = st.columns(2)

with col1:
    if st.button("🔄 Use Home Input", use_container_width=True):
        st.session_state.qa_context = st.session_state.get("input_text", "")

with col2:
    if st.button("📋 Load Example", use_container_width=True):
        st.session_state.qa_context = example_context
        st.session_state.qa_question = example_question

# -----------------------------------
# User Input
# -----------------------------------
context = st.text_area(
    "📄 Context",
    key="qa_context",
    height=250,
    placeholder="Paste your context paragraph here..."
)

question = st.text_input(
    "❓ Question",
    key="qa_question",
    placeholder="Ask a question based on the context..."
)

# -----------------------------------
# Answer Button
# -----------------------------------
if st.button("🚀 Get Answer", use_container_width=True):

    if context.strip() == "" or question.strip() == "":
        st.warning("Please provide both context and question.")

    else:

        with st.spinner("Finding answer..."):

            result = qa_pipeline(
                question=question,
                context=context
            )

        st.success("Answer Found!")

        st.subheader("Answer")

        st.info(result["answer"])

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Confidence",
                f"{result['score']:.2%}"
            )

        with col2:
            st.metric(
                "Context Words",
                len(context.split())
            )

        st.download_button(
            label="📥 Download Answer",
            data=f"""
Question:
{question}

Answer:
{result['answer']}

Confidence:
{result['score']:.2%}
""",
            file_name="answer.txt",
            mime="text/plain"
        )

# -----------------------------------
# Navigation
# -----------------------------------
st.divider()

col1, col2 = st.columns(2)

with col1:
    if st.button("🏠 Home", use_container_width=True):
        st.switch_page("Home.py")

with col2:
    if st.button("🌍 Translator ➜", use_container_width=True):
        st.switch_page("pages/4_🌍_Translator.py")
