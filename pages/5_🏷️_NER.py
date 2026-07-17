import streamlit as st
import pandas as pd
from utils.models import load_ner_model

# ----------------------------------
# Page Configuration
# ----------------------------------
st.set_page_config(
    page_title="Named Entity Recognition",
    page_icon="🏷️",
    layout="wide"
)

ner = load_ner_model()

# ----------------------------------
# Header
# ----------------------------------
st.title("🏷️ Named Entity Recognition (NER)")
st.write(
    "Identify named entities such as people, organizations, locations, dates, and more."
)

# ----------------------------------
# Sidebar
# ----------------------------------
st.sidebar.header("About")

st.sidebar.info("""
Named Entity Recognition (NER) identifies important entities in text.

Examples:
- 👤 Person
- 🏢 Organization
- 📍 Location
- 📅 Date
- 💰 Money
""")

# ----------------------------------
# Example
# ----------------------------------
example = """
Sundar Pichai is the CEO of Google. He met Prime Minister Narendra Modi in New Delhi
to discuss Artificial Intelligence initiatives for India.
"""

if st.button("📋 Load Example"):
    st.session_state.ner_text = example

# ----------------------------------
# Input
# ----------------------------------
text = st.text_area(
    "Enter Text",
    value=st.session_state.get("ner_text", ""),
    height=220,
    placeholder="Type or paste your text..."
)

# ----------------------------------
# Extract
# ----------------------------------
if st.button("🔍 Extract Entities", use_container_width=True):

    if text.strip() == "":
        st.warning("Please enter some text.")

    else:

        with st.spinner("Extracting entities..."):

            results = ner(text)

        st.success("Entities Extracted Successfully!")

        if len(results) == 0:
            st.info("No entities found.")

        else:

            data = []

            for entity in results:

                data.append({
                    "Entity": entity["word"],
                    "Type": entity["entity_group"],
                    "Confidence": f"{entity['score']:.2%}"
                })

            df = pd.DataFrame(data)

            st.subheader("Detected Entities")

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )

            st.metric(
                "Total Entities",
                len(df)
            )

            st.download_button(
                "📥 Download Results",
                df.to_csv(index=False),
                file_name="entities.csv",
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
    if st.button("🎯 Zero-Shot ➜", use_container_width=True):
        st.switch_page("pages/6_🎯_ZeroShot.py")
