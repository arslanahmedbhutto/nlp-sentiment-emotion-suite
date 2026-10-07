"""Text cleaning."""

import re

_HTML_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"http\S+|www\.\S+")
_NON_LETTER = re.compile(r"[^a-z\s]")
_SPACES = re.compile(r"\s+")


def clean_review(text: str) -> str:
    """Light cleaning for TF-IDF models.

    Keeps stop words and negations ("not", "didn't" -> "didnt"), because
    words like "not" flip sentiment and n-grams such as "not good" need them.
    """
    text = _HTML_TAG.sub(" ", text.lower())
    text = _URL.sub(" ", text)
    text = text.replace("'", "")
    text = _NON_LETTER.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


# Stop-word list and cleaning used when the BiLSTM was trained
# (notebooks/02_emotion_bilstm_attention.ipynb). Inference must use the
# exact same function, so it is kept verbatim.
BILSTM_STOP_WORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
    "yourself", "he", "him", "his", "himself", "she", "her", "hers", "they", "them",
    "their", "theirs", "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "should", "shall", "can", "could", "may",
    "might", "must", "the", "a", "an", "and", "but", "or", "nor", "for", "yet", "so",
    "at", "by", "in", "of", "on", "to", "up", "as", "if", "it", "its", "than", "then",
    "no", "not", "only", "own", "same", "too", "very", "s", "t", "just", "don",
    "now", "d", "ll", "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn",
    "hadn", "hasn", "haven", "isn", "ma", "mightn", "mustn", "needn", "shan",
    "shouldn", "wasn", "weren", "won", "wouldn",
}


def clean_for_bilstm(text: str) -> str:
    text = text.lower()
    text = _URL.sub("", text)
    text = _NON_LETTER.sub(" ", text)
    text = _SPACES.sub(" ", text).strip()
    tokens = [w for w in text.split() if w not in BILSTM_STOP_WORDS and len(w) > 1]
    return " ".join(tokens)
