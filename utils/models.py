import streamlit as st
from transformers import pipeline


# -------------------------------
# Sentiment Analysis
# -------------------------------
@st.cache_resource
def load_sentiment_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )


# -------------------------------
# Text Summarization
# -------------------------------
@st.cache_resource
def load_summarizer():
    return pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6"
    )


# -------------------------------
# Question Answering
# -------------------------------
@st.cache_resource
def load_qa_model():
    return pipeline(
        "question-answering",
        model="distilbert-base-cased-distilled-squad"
    )


# -------------------------------
# Translation (English → French)
# -------------------------------
@st.cache_resource
def load_summarizer():
    return pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6"
    )

# -------------------------------
# Named Entity Recognition
# -------------------------------
@st.cache_resource
def load_ner_model():
    return pipeline(
        "ner",
        model="dslim/bert-base-NER",
        aggregation_strategy="simple"
    )


# -------------------------------
# Zero-Shot Classification
# -------------------------------
@st.cache_resource
def load_zero_shot():
    return pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli"
    )
