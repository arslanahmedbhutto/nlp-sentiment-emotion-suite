import pytest

from nlp_suite.data import EMOTION_LABELS, load_emotion_split, load_imdb
from nlp_suite.text import clean_for_bilstm, clean_review


def test_clean_review_strips_html_and_keeps_negation():
    assert clean_review("It was <br /><br />NOT good!!") == "it was not good"
    assert clean_review("I didn't like it") == "i didnt like it"
    assert clean_review("see http://example.com now") == "see now"


def test_clean_for_bilstm_matches_training_notebook():
    # Example printed in notebooks/02_emotion_bilstm_attention.ipynb
    raw = "i can go from feeling so hopeless to so damned hopeful just from being around someone who cares and is awake"
    assert clean_for_bilstm(raw) == "go from feeling hopeless damned hopeful from around someone cares awake"


def test_emotion_splits_load():
    for split, n in [("train", 16000), ("val", 2000), ("test", 2000)]:
        df = load_emotion_split(split)
        assert len(df) == n
        assert set(df["label"]) == set(EMOTION_LABELS)
        assert df["text"].str.len().min() > 0


def test_emotion_loader_parses_text_with_semicolons(tmp_path):
    (tmp_path / "train.txt").write_text("a; b; c;joy\n\nhello;fear\n")
    df = load_emotion_split("train", data_dir=tmp_path)
    assert df.to_dict("list") == {"text": ["a; b; c", "hello"], "label": ["joy", "fear"]}


def test_missing_imdb_gives_download_hint(tmp_path):
    with pytest.raises(FileNotFoundError, match="kaggle"):
        load_imdb(tmp_path / "nope.csv")
