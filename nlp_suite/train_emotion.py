"""Compare classical TF-IDF models with the BiLSTM + attention model on 6 emotions.

    python -m nlp_suite.train_emotion

Classical models are fitted on `train`, chosen by macro-F1 on `val`, and
reported on `test`. The BiLSTM (trained in the notebook, early-stopped on
`val`) is evaluated on the same `test` split when TensorFlow is installed.
"""

import time
from datetime import datetime, timezone

import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from nlp_suite import FIGURES_DIR, MODELS_DIR, REPORTS_DIR
from nlp_suite.data import EMOTION_LABELS, load_emotion_split
from nlp_suite.models import TextClassifier, emotion_candidates
from nlp_suite.reporting import (bar_comparison, confusion_matrices, grouped_bars,
                                 markdown_table, write_json)

BILSTM_NAME = "BiLSTM + Attention"


def scores(y_true, y_pred) -> dict:
    per_class = f1_score(y_true, y_pred, labels=EMOTION_LABELS, average=None)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro F1": f1_score(y_true, y_pred, average="macro"),
        "Weighted F1": f1_score(y_true, y_pred, average="weighted"),
        "per_class_f1": dict(zip(EMOTION_LABELS, map(float, per_class))),
    }


def main():
    train, val, test = (load_emotion_split(s) for s in ("train", "val", "test"))
    print(f"Emotions: train {len(train):,} | val {len(val):,} | test {len(test):,}")

    rows, preds, fitted = [], {}, {}
    for name, pipe in emotion_candidates().items():
        t0 = time.time()
        pipe.fit(train["text"], train["label"])
        assert list(pipe.classes_) == EMOTION_LABELS
        val_f1 = f1_score(val["label"], pipe.predict(val["text"]), average="macro")
        pred = pipe.predict(test["text"])
        rows.append({"Model": name, "Type": "TF-IDF", "Val Macro F1": float(val_f1), **scores(test["label"], pred)})
        preds[name], fitted[name] = pred, pipe
        print(f"  {name:<22} val macro-F1 {val_f1:.4f}  test acc {rows[-1]['Accuracy']:.4f}  ({time.time() - t0:.0f}s)")

    classical = [r for r in rows]
    best_classic = max(classical, key=lambda r: r["Val Macro F1"])["Model"]

    try:
        from nlp_suite.bilstm import BiLSTMEmotion
        bilstm = BiLSTMEmotion()
        probs = bilstm.predict_proba_batch(test["text"].tolist())
        pred = [bilstm.labels[i] for i in probs.argmax(axis=1)]
        val_pred = [bilstm.labels[i] for i in bilstm.predict_proba_batch(val["text"].tolist()).argmax(axis=1)]
        rows.append({"Model": BILSTM_NAME, "Type": "Deep learning",
                     "Val Macro F1": float(f1_score(val["label"], val_pred, average="macro")),
                     **scores(test["label"], pred)})
        preds[BILSTM_NAME] = pred
        print(f"  {BILSTM_NAME:<22} test acc {rows[-1]['Accuracy']:.4f}")
    except ImportError:
        print("  TensorFlow not installed: skipping BiLSTM evaluation")

    rows.sort(key=lambda r: r["Macro F1"], reverse=True)

    MODELS_DIR.mkdir(exist_ok=True)
    app_models = {n: TextClassifier(p, EMOTION_LABELS, next(r for r in rows if r["Model"] == n))
                  for n, p in fitted.items()}
    joblib.dump({"models": app_models, "best": best_classic}, MODELS_DIR / "emotion_classic.joblib", compress=3)

    write_json({
        "dataset": "Emotions dataset for NLP (train/val/test)",
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "selection": "classical models chosen by macro-F1 on val",
        "best_classical": best_classic,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "results": rows,
    }, REPORTS_DIR / "emotion_metrics.json")

    cols = ["Model", "Type", "Accuracy", "Macro F1", "Weighted F1"]
    per_class_rows = [{"Model": r["Model"], **{e: r["per_class_f1"][e] for e in EMOTION_LABELS}} for r in rows]
    (REPORTS_DIR / "emotion_comparison.md").write_text(
        "# Emotion classification: model comparison (test set, 2,000 texts)\n\n"
        + markdown_table(rows, cols, bold_row=rows[0]["Model"])
        + "\n\n## F1 per emotion\n\n" + markdown_table(per_class_rows, ["Model"] + EMOTION_LABELS) + "\n"
    )

    bar_comparison({r["Model"]: r["Macro F1"] for r in rows}, rows[0]["Model"],
                   "Emotion classification: macro-F1 by model", "Macro F1 (2K test texts, 6 classes)",
                   FIGURES_DIR / "emotion_model_comparison.png", xlim=(0.6, 0.95))

    compare = [best_classic] + ([BILSTM_NAME] if BILSTM_NAME in preds else [])
    grouped_bars(EMOTION_LABELS,
                 {n: [next(r for r in rows if r["Model"] == n)["per_class_f1"][e] for e in EMOTION_LABELS]
                  for n in compare},
                 "F1 per emotion", "F1 score", FIGURES_DIR / "emotion_per_class_f1.png")
    confusion_matrices({n: confusion_matrix(test["label"], preds[n], labels=EMOTION_LABELS) for n in compare},
                       EMOTION_LABELS, FIGURES_DIR / "emotion_confusion_matrices.png")


if __name__ == "__main__":
    main()
