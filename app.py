"""
Mini AI Toolkit — Streamlit app
Pipeline: User Input -> Choose Task -> Hugging Face Pipeline -> Prediction -> Output

Run with:
    pip install streamlit transformers torch --break-system-packages
    streamlit run mini_ai_toolkit.py
"""

import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Mini AI Toolkit", page_icon="🧰", layout="centered")

# ---------------------------------------------------------------------------
# Task configuration: maps each task to a Hugging Face pipeline task name
# and (optionally) a specific model checkpoint.
# ---------------------------------------------------------------------------
TASKS = {
    "Sentiment analysis": {
        "pipeline_task": "sentiment-analysis",
        "model": None,  # use default checkpoint
        "icon": "😊",
        "description": "Reads your text and tells you if it's positive or negative.",
    },
    "Text generation": {
        "pipeline_task": "text-generation",
        "model": "gpt2",
        "icon": "✍️",
        "description": "Continues your prompt with AI-generated text.",
    },
    "Summarization": {
        "pipeline_task": "summarization",
        "model": "sshleifer/distilbart-cnn-12-6",
        "icon": "📝",
        "description": "Condenses a long passage into a short summary.",
    },
    "Question answering": {
        "pipeline_task": "question-answering",
        "model": None,
        "icon": "❓",
        "description": "Finds the answer to your question inside a given passage.",
    },
    "Translation (EN to FR)": {
        "pipeline_task": "translation_en_to_fr",
        "model": None,
        "icon": "🌐",
        "description": "Translates your English text into French.",
    },
    "Named entity recognition (NER)": {
        "pipeline_task": "ner",
        "model": None,
        "icon": "🏷️",
        "description": "Detects names of people, places, and organizations in your text.",
    },
    "Zero-shot classification": {
        "pipeline_task": "zero-shot-classification",
        "model": None,
        "icon": "🎯",
        "description": "Sorts your text into categories you define, with no training needed.",
    },
}


# ---------------------------------------------------------------------------
# Cache loaded pipelines so switching tasks doesn't reload the model every run
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_pipeline(task_name: str):
    config = TASKS[task_name]
    kwargs = {"task": config["pipeline_task"]}
    if config["model"]:
        kwargs["model"] = config["model"]
    return pipeline(**kwargs)


def format_output(task_name: str, result):
    """Task-aware formatting of raw model output into readable text."""
    if task_name == "Sentiment analysis":
        r = result[0]
        return f"**{r['label']}** (confidence: {r['score']:.2%})"

    if task_name == "Text generation":
        return result[0]["generated_text"]

    if task_name == "Summarization":
        return result[0]["summary_text"]

    if task_name == "Question answering":
        r = result
        return f"**Answer:** {r['answer']}  \n(confidence: {r['score']:.2%})"

    if task_name == "Translation (EN to FR)":
        return result[0]["translation_text"]

    if task_name == "Named entity recognition (NER)":
        if not result:
            return "No entities found."
        lines = [f"- **{e['word']}** → {e['entity']} (score: {e['score']:.2%})" for e in result]
        return "\n".join(lines)

    if task_name == "Zero-shot classification":
        lines = [f"- **{label}**: {score:.2%}" for label, score in zip(result["labels"], result["scores"])]
        return "\n".join(lines)

    return str(result)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🧰 Mini AI Toolkit")
st.caption("Pick a task, type your text, and see what the model does with it.")

st.markdown(
    """
    <style>
    div[data-testid="stButton"] > button {
        width: 100%;
        height: 100%;
        border-radius: 12px;
        padding: 1rem 0.75rem;
        text-align: left;
        white-space: normal;
        line-height: 1.4;
    }
    div[data-testid="stButton"] > button p {
        white-space: normal;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "selected_task" not in st.session_state:
    st.session_state.selected_task = list(TASKS.keys())[0]

st.subheader("1. Choose a task")

task_items = list(TASKS.items())
cols_per_row = 3
for row_start in range(0, len(task_items), cols_per_row):
    row_items = task_items[row_start: row_start + cols_per_row]
    cols = st.columns(cols_per_row)
    for col, (name, cfg) in zip(cols, row_items):
        with col:
            selected = st.session_state.selected_task == name
            label = f"{cfg['icon']}  **{name}**\n\n{cfg['description']}"
            btn_type = "primary" if selected else "secondary"
            if st.button(label, key=f"card_{name}", type=btn_type, use_container_width=True):
                st.session_state.selected_task = name
                st.rerun()

task_name = st.session_state.selected_task

st.divider()
st.subheader("2. Enter your text")
st.caption(f"Selected task: **{TASKS[task_name]['icon']} {task_name}** — {TASKS[task_name]['description']}")

user_input = st.text_area(
    "Text prompt or question",
    height=140,
    placeholder="Type something here...",
    label_visibility="collapsed",
)

# Extra fields required by specific tasks
context = None
candidate_labels = None

if task_name == "Question answering":
    context = st.text_area("Context (the passage to search for the answer)", height=140)

if task_name == "Zero-shot classification":
    candidate_labels_raw = st.text_input(
        "Candidate labels (comma-separated)", placeholder="e.g. sports, politics, technology"
    )
    candidate_labels = [l.strip() for l in candidate_labels_raw.split(",") if l.strip()]

run = st.button("Run", type="primary")

if run:
    if not user_input.strip():
        st.warning("Please enter some text first.")
    elif task_name == "Question answering" and not context:
        st.warning("Please provide context for question answering.")
    elif task_name == "Zero-shot classification" and not candidate_labels:
        st.warning("Please provide at least one candidate label.")
    else:
        with st.spinner(f"Running {task_name.lower()}..."):
            nlp = load_pipeline(task_name)

            if task_name == "Question answering":
                result = nlp(question=user_input, context=context)
            elif task_name == "Zero-shot classification":
                result = nlp(user_input, candidate_labels=candidate_labels)
            elif task_name == "Text generation":
                result = nlp(user_input, max_new_tokens=60, num_return_sequences=1)
            else:
                result = nlp(user_input)

        st.subheader("Prediction")
        st.markdown(format_output(task_name, result))

        with st.expander("Raw output"):
            st.write(result)
