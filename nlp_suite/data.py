"""Dataset loading."""

from pathlib import Path

import pandas as pd

from nlp_suite import DATA_DIR

IMDB_PATH = DATA_DIR / "imdb" / "IMDB Dataset.csv"
EMOTION_DIR = DATA_DIR / "emotions"
EMOTION_LABELS = ["anger", "fear", "joy", "love", "sadness", "surprise"]


def load_imdb(path: Path = IMDB_PATH) -> pd.DataFrame:
    """IMDb 50K reviews -> columns: text, label (1 = positive, 0 = negative)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download 'IMDB Dataset.csv' from "
            "https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews "
            "and put it in data/imdb/."
        )
    df = pd.read_csv(path)
    df = df.drop_duplicates(subset="review").reset_index(drop=True)
    return pd.DataFrame({
        "text": df["review"],
        "label": (df["sentiment"] == "positive").astype(int),
    })


def load_emotion_split(split: str, data_dir: Path = EMOTION_DIR) -> pd.DataFrame:
    """Read one split ('train', 'val', 'test') of `text;label` lines."""
    texts, labels = [], []
    with open(Path(data_dir) / f"{split}.txt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if ";" in line:
                text, label = line.rsplit(";", 1)
                texts.append(text.strip())
                labels.append(label.strip())
    return pd.DataFrame({"text": texts, "label": labels})
