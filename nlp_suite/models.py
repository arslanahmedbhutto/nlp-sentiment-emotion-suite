"""Classical TF-IDF models, plus per-word explanations for linear models."""

from dataclasses import dataclass, field

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from nlp_suite.text import clean_review

SEED = 42


def tfidf(**overrides) -> TfidfVectorizer:
    params = dict(
        preprocessor=clean_review,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
        sublinear_tf=True,
        max_features=100_000,
    )
    params.update(overrides)
    return TfidfVectorizer(**params)


def sentiment_candidates() -> dict[str, Pipeline]:
    """Models compared on IMDb. The first one reproduces the original course notebook."""
    return {
        "Course baseline (TF-IDF 5k + LR)": Pipeline([
            ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
            ("clf", LogisticRegression(max_iter=1000)),
        ]),
        "Naive Bayes": Pipeline([
            ("tfidf", tfidf()),
            ("clf", MultinomialNB(alpha=0.1)),
        ]),
        "Logistic Regression": Pipeline([
            ("tfidf", tfidf()),
            ("clf", LogisticRegression(C=4.0, max_iter=2000)),
        ]),
        "Linear SVM": Pipeline([
            ("tfidf", tfidf()),
            ("clf", CalibratedClassifierCV(LinearSVC(C=0.5), cv=3)),
        ]),
    }


def emotion_candidates() -> dict[str, Pipeline]:
    """Models compared on the 6-class emotion dataset (short texts)."""
    small = dict(min_df=2, max_features=50_000)
    return {
        "Naive Bayes": Pipeline([
            ("tfidf", tfidf(**small)),
            ("clf", MultinomialNB(alpha=0.1)),
        ]),
        "Logistic Regression": Pipeline([
            ("tfidf", tfidf(**small)),
            ("clf", LogisticRegression(C=10.0, max_iter=3000, class_weight="balanced")),
        ]),
        "Linear SVM": Pipeline([
            ("tfidf", tfidf(**small)),
            ("clf", CalibratedClassifierCV(LinearSVC(C=0.5, class_weight="balanced"), cv=3)),
        ]),
    }


def linear_weights(clf) -> np.ndarray:
    """Per-class feature weights, shape (n_classes, n_features).

    For binary models one row is returned per class (the second is the
    negation of the first), so callers can always index by class.
    """
    if isinstance(clf, CalibratedClassifierCV):
        w = np.mean([c.estimator.coef_ for c in clf.calibrated_classifiers_], axis=0)
    elif isinstance(clf, MultinomialNB):
        lp = clf.feature_log_prob_
        w = lp[1:] - lp[:1] if lp.shape[0] == 2 else lp - lp.mean(axis=0)
    else:
        w = clf.coef_
    if w.shape[0] == 1:
        w = np.vstack([-w[0], w[0]])
    return w


@dataclass
class TextClassifier:
    """A fitted TF-IDF pipeline with readable class names."""

    pipeline: Pipeline
    labels: list[str]
    metrics: dict = field(default_factory=dict)

    def predict_proba(self, text: str) -> dict[str, float]:
        probs = self.pipeline.predict_proba([text])[0]
        return {self.labels[i]: float(p) for i, p in enumerate(probs)}

    def predict(self, text: str) -> str:
        probs = self.predict_proba(text)
        return max(probs, key=probs.get)

    def explain(self, text: str, label: str, k: int = 8) -> list[tuple[str, float]]:
        """Words/bigrams in `text` that push hardest toward (+) or away from (-) `label`."""
        vec = self.pipeline.named_steps["tfidf"]
        x = vec.transform([text])
        if x.nnz == 0:
            return []
        weights = linear_weights(self.pipeline.named_steps["clf"])[self.labels.index(label)]
        names = vec.get_feature_names_out()
        contrib = {names[j]: float(v * weights[j]) for j, v in zip(x.indices, x.data)}
        ranked = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)
        return ranked[:k]
