import streamlit as st

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Mini AI Toolkit",
    page_icon="🤖",
    layout="wide"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.title {
    text-align: center;
    font-size: 48px;
    font-weight: bold;
    color: #2563EB;
}

.subtitle {
    text-align: center;
    color: gray;
    font-size: 18px;
    margin-bottom: 30px;
}

.card {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}

.footer {
    text-align: center;
    color: gray;
    margin-top: 30px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown('<p class="title">🤖 Mini AI Toolkit</p>', unsafe_allow_html=True)
st.subheader("📝 Enter Your Text")

user_text = st.text_area(
    "Input",
    value=st.session_state.get("input_text", ""),
    height=250,
    placeholder="Type or paste your text here..."
)

st.markdown(
    '<p class="subtitle">Choose an NLP task to get started</p>',
    unsafe_allow_html=True
)

# -----------------------------
# First Row
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.subheader("😊 Sentiment Analysis")
        st.write("Analyze whether text is Positive, Negative or Neutral.")
        def open_task(page):
    if not user_text.strip():
        st.warning("Please enter some text first.")
    else:
        st.session_state["input_text"] = user_text
        st.switch_page(page)

with col2:
    with st.container(border=True):
        st.subheader("📝 Text Summarizer")
        st.write("Generate concise summaries from long articles.")
        if st.button("Open", use_container_width=True, key="summary"):
            st.switch_page("pages/2_📝_Summarizer.py")

# -----------------------------
# Second Row
# -----------------------------
col3, col4 = st.columns(2)

with col3:
    with st.container(border=True):
        st.subheader("💬 Question Answering")
        st.write("Ask questions based on a given context.")
        if st.button("Open", use_container_width=True, key="qa"):
            st.switch_page("pages/3_💬_Question_Answering.py")

with col4:
    with st.container(border=True):
        st.subheader("🌍 Language Translation")
        st.write("Translate English text into another language.")
        if st.button("Open", use_container_width=True, key="translation"):
            st.switch_page("pages/4_🌍_Translator.py")

# -----------------------------
# Third Row
# -----------------------------
col5, col6 = st.columns(2)

with col5:
    with st.container(border=True):
        st.subheader("🏷️ Named Entity Recognition")
        st.write("Identify people, organizations and locations.")
        if st.button("Open", use_container_width=True, key="ner"):
            st.switch_page("pages/5_🏷️_NER.py")

with col6:
    with st.container(border=True):
        st.subheader("🎯 Zero-Shot Classification")
        st.write("Classify text into custom labels without training.")
        if st.button("Open", use_container_width=True, key="zero"):
            st.switch_page("pages/6_🎯_ZeroShot.py")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")

st.markdown(
    '<p class="footer">🚀 Built with Streamlit & Hugging Face Transformers</p>',
    unsafe_allow_html=True
)
