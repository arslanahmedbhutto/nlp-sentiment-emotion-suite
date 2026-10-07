import joblib
import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from nlp_suite import MODELS_DIR
from nlp_suite.data import EMOTION_LABELS
from nlp_suite.models import TextClassifier, emotion_candidates, linear_weights, sentiment_candidates

TOY_TEXTS = ["great fun movie", "wonderful great acting", "loved it great",
             "terrible boring movie", "awful boring plot", "hated it terrible"] * 5
TOY_LABELS = [1, 1, 1, 0, 0, 0] * 5


@pytest.mark.parametrize("name", [n for n in sentiment_candidates() if not n.startswith("Course")])
def test_candidates_fit_and_explain(name):
    pipe: Pipeline = sentiment_candidates()[name]
    pipe.set_params(tfidf__min_df=1)
    pipe.fit(TOY_TEXTS, TOY_LABELS)
    clf = TextClassifier(pipe, ["negative", "positive"])

    assert clf.predict("a great film") == "positive"
    probs = clf.predict_proba("so boring")
    assert probs["negative"] > probs["positive"]
    assert abs(sum(probs.values()) - 1) < 1e-6

    terms = dict(clf.explain("great but boring", "positive"))
    assert terms["great"] > 0 > terms["boring"]


def test_explain_unknown_words_is_empty():
    pipe = sentiment_candidates()["Logistic Regression"].set_params(tfidf__min_df=1)
    pipe.fit(TOY_TEXTS, TOY_LABELS)
    assert TextClassifier(pipe, ["negative", "positive"]).explain("zzz qqq", "positive") == []


def test_linear_weights_shape_for_multiclass():
    pipe = emotion_candidates()["Naive Bayes"].set_params(tfidf__min_df=1)
    texts = ["happy joy", "sad cry", "angry mad", "scared afraid", "love dear", "wow shocked"]
    pipe.fit(texts, ["joy", "sadness", "anger", "fear", "love", "surprise"])
    w = linear_weights(pipe.named_steps["clf"])
    assert w.shape == (6, len(pipe.named_steps["tfidf"].vocabulary_))


# ---------- saved models (committed in models/) ----------

@pytest.fixture(scope="module")
def sentiment_bundle():
    return joblib.load(MODELS_DIR / "sentiment.joblib")


@pytest.fixture(scope="module")
def emotion_bundle():
    return joblib.load(MODELS_DIR / "emotion_classic.joblib")


def test_saved_sentiment_models(sentiment_bundle):
    assert sentiment_bundle["best"] in sentiment_bundle["models"]
    for clf in sentiment_bundle["models"].values():
        assert clf.predict("An absolute masterpiece, brilliant acting and a moving story.") == "positive"
        assert clf.predict("A boring, badly written waste of two hours.") == "negative"
        assert clf.metrics["Test Acc"] > 0.85


def test_saved_emotion_models(emotion_bundle):
    for clf in emotion_bundle["models"].values():
        assert clf.labels == EMOTION_LABELS
        assert clf.predict("i feel so happy and cheerful today") == "joy"
        assert clf.predict("i feel scared and terrified") == "fear"


def test_bilstm_loads_and_predicts():
    pytest.importorskip("tensorflow")
    from nlp_suite.bilstm import BiLSTMEmotion

    m = BiLSTMEmotion()
    probs = m.predict_proba("i feel so happy and cheerful today")
    assert max(probs, key=probs.get) == "joy"
    att = m.attention("i feel so happy and cheerful today")
    assert [w for w, _ in att] == ["feel", "happy", "cheerful", "today"]
    assert np.isclose(sum(a for _, a in att), 1.0)


def test_bilstm_word_importance_finds_the_emotion_word():
    pytest.importorskip("tensorflow")
    from nlp_suite.bilstm import BiLSTMEmotion

    imp = dict(BiLSTMEmotion().word_importance("I feel scared to be alone in the house", "fear"))
    assert max(imp, key=imp.get) == "scared"
