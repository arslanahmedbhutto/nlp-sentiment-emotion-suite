"""Streamlit demo: sentiment (IMDb) and emotion classification, with explanations.

    streamlit run app/streamlit_app.py
"""

import html
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
import streamlit as st

from nlp_suite import FIGURES_DIR, MODELS_DIR, REPORTS_DIR

SUPPORT, AGAINST = "#2a78d6", "#e34948"
EMOJI = {"anger": "😠", "fear": "😨", "joy": "😄", "love": "❤️", "sadness": "😢", "surprise": "😲",
         "positive": "👍", "negative": "👎"}

SENTIMENT_EXAMPLES = {
    "Write your own…": "",
    "Glowing review": "An absolute masterpiece. The acting is superb, the story is moving and I was hooked from the first minute.",
    "Negation": "I expected to hate it, but honestly it was not bad at all. Not the best film, yet surprisingly enjoyable.",
    "Disappointed fan": "I loved the first two films, but this one was a boring, predictable mess with terrible dialogue.",
}
EMOTION_EXAMPLES = {
    "Write your own…": "",
    "Joy": "I just got the job offer and I feel so happy and grateful today!",
    "Fear": "I keep hearing noises outside and I feel scared to be alone in the house.",
    "Anger": "I am so annoyed that they cancelled my order again without telling me.",
    "Surprise": "I was shocked to find out they had planned a party for me, I had no idea!",
}

st.set_page_config(page_title="NLP Sentiment & Emotion Suite", page_icon="🧠", layout="wide")


# ---------- loading ----------

@st.cache_resource
def load_classic(name: str):
    return joblib.load(MODELS_DIR / name)


@st.cache_resource
def load_bilstm():
    try:
        from nlp_suite.bilstm import BiLSTMEmotion
        return BiLSTMEmotion()
    except ImportError:
        return None


# ---------- rendering helpers ----------

def prob_chart(probs: dict[str, float]):
    df = pd.DataFrame({"label": list(probs), "probability": list(probs.values())})
    df = df.sort_values("probability", ascending=False).set_index("label")
    st.bar_chart(df, horizontal=True, color=SUPPORT, height=40 + 34 * len(df))


def contribution_chips(contribs: list[tuple[str, float]],
                       how: str = "score = TF-IDF value × model weight"):
    if not contribs:
        st.caption("None of the words in this text are in the model's vocabulary.")
        return
    top = max(abs(v) for _, v in contribs) or 1.0
    # Hide words whose effect is negligible next to the strongest one
    contribs = [(t, v) for t, v in contribs if abs(v) >= 0.02 * top]
    chips = []
    for term, v in contribs:
        color = SUPPORT if v > 0 else AGAINST
        alpha = int(40 + 180 * abs(v) / top)
        chips.append(
            f'<span style="display:inline-block;margin:3px;padding:3px 10px;border-radius:12px;'
            f'background:{color}{alpha:02x};border:1px solid {color}">'
            f'{html.escape(term)} <small>{v:+.2f}</small></span>'
        )
    st.markdown("".join(chips), unsafe_allow_html=True)
    st.caption(f'<span style="color:{SUPPORT}">■</span> pushes toward the prediction · '
               f'<span style="color:{AGAINST}">■</span> pushes against it · '
               + how, unsafe_allow_html=True)


def text_input(examples: dict[str, str], key: str) -> str:
    choice = st.selectbox("Example", list(examples), key=f"{key}_ex")
    return st.text_area("Text", value=examples[choice], height=120, key=f"{key}_{choice}",
                        placeholder="Type or paste some text…")


def headline(label: str, conf: float):
    st.metric("Prediction", f"{EMOJI.get(label, '')} {label.capitalize()}", f"{conf:.1%} confidence",
              delta_color="off")


# ---------- page ----------

st.title("🧠 NLP Sentiment & Emotion Suite")
st.write("Classify movie-review **sentiment** and the **emotion** in a sentence, and see *why* the model decided.")

tab_sent, tab_emo, tab_cmp = st.tabs(["🎬 Sentiment", "🎭 Emotion", "📊 Model comparison"])

with tab_sent:
    bundle = load_classic("sentiment.joblib")
    names = list(bundle["models"])
    model_name = st.selectbox("Model", names, index=names.index(bundle["best"]), key="sent_model")
    clf = bundle["models"][model_name]
    st.caption(f"Test accuracy on 10K IMDb reviews: **{clf.metrics['Test Acc']:.1%}**")
    text = text_input(SENTIMENT_EXAMPLES, "sent")
    if text.strip():
        probs = clf.predict_proba(text)
        label = max(probs, key=probs.get)
        c1, c2 = st.columns([1, 2])
        with c1:
            headline(label, probs[label])
            prob_chart(probs)
        with c2:
            st.subheader("Why?")
            contribution_chips(clf.explain(text, label, k=12))

with tab_emo:
    bundle = load_classic("emotion_classic.joblib")
    bilstm = load_bilstm()
    names = list(bundle["models"]) + (["BiLSTM + Attention"] if bilstm else [])
    model_name = st.selectbox("Model", names, index=names.index(bundle["best"]), key="emo_model")
    if not bilstm:
        st.caption("Install TensorFlow (`pip install -r requirements-dl.txt`) to enable the BiLSTM model.")
    text = text_input(EMOTION_EXAMPLES, "emo")
    if text.strip():
        if model_name == "BiLSTM + Attention":
            probs = bilstm.predict_proba(text)
        else:
            clf = bundle["models"][model_name]
            probs = clf.predict_proba(text)
        label = max(probs, key=probs.get)
        c1, c2 = st.columns([1, 2])
        with c1:
            headline(label, probs[label])
            prob_chart(probs)
        with c2:
            st.subheader("Why?")
            if model_name == "BiLSTM + Attention":
                importance = sorted(bilstm.word_importance(text, label), key=lambda kv: abs(kv[1]), reverse=True)
                contribution_chips(importance[:12],
                                   "score = drop in confidence when the word is removed "
                                   "(words after the BiLSTM's preprocessing)")
            else:
                contribution_chips(clf.explain(text, label, k=12))

with tab_cmp:
    for task, md, figs in [
        ("Sentiment (IMDb)", "sentiment_comparison.md", ["sentiment_model_comparison.png", "sentiment_confusion_matrix.png"]),
        ("Emotion", "emotion_comparison.md", ["emotion_model_comparison.png", "emotion_per_class_f1.png",
                                              "emotion_confusion_matrices.png"]),
    ]:
        st.header(task)
        path = REPORTS_DIR / md
        if path.exists():
            st.markdown(path.read_text().split("\n", 2)[2])
        for f in figs:
            if (FIGURES_DIR / f).exists():
                st.image(str(FIGURES_DIR / f))

st.caption("Educational project. Predictions can be wrong, especially for sarcasm or text unlike the training data.")
