"""Compare TF-IDF models on IMDb reviews and save the fitted models.

    python -m nlp_suite.train_sentiment
"""

import time
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from nlp_suite import FIGURES_DIR, MODELS_DIR, REPORTS_DIR
from nlp_suite.data import load_imdb
from nlp_suite.models import SEED, TextClassifier, sentiment_candidates
from nlp_suite.reporting import bar_comparison, confusion_matrices, markdown_table, write_json

LABELS = ["negative", "positive"]
CV_FOLDS = 3


def main():
    df = load_imdb()
    print(f"IMDb: {len(df):,} unique reviews")
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, stratify=df["label"], random_state=SEED
    )
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED)

    rows, fitted, preds = [], {}, {}
    for name, pipe in sentiment_candidates().items():
        t0 = time.time()
        cv_acc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        rows.append({
            "Model": name,
            "CV Acc": float(cv_acc.mean()),
            "CV Std": float(cv_acc.std()),
            "Test Acc": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
            "F1": f1_score(y_test, pred),
            "ROC-AUC": roc_auc_score(y_test, proba),
        })
        fitted[name], preds[name] = pipe, pred
        print(f"  {name:<34} CV acc {cv_acc.mean():.4f} ± {cv_acc.std():.4f}   "
              f"test acc {rows[-1]['Test Acc']:.4f}   ({time.time() - t0:.0f}s)")

    rows.sort(key=lambda r: r["CV Acc"], reverse=True)
    best = rows[0]["Model"]
    print(f"Best by CV accuracy: {best}")

    # The app offers the three tuned models; the course baseline is only reported.
    app_models = {
        name: TextClassifier(pipe, LABELS, next(r for r in rows if r["Model"] == name))
        for name, pipe in fitted.items() if not name.startswith("Course baseline")
    }
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"models": app_models, "best": best}, MODELS_DIR / "sentiment.joblib", compress=3)

    write_json({
        "dataset": "IMDb 50K movie reviews (duplicates removed)",
        "n_train": len(X_train), "n_test": len(X_test),
        "cv_folds": CV_FOLDS, "selection": "mean CV accuracy",
        "best_model": best,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "results": rows,
    }, REPORTS_DIR / "sentiment_metrics.json")
    cols = ["Model", "CV Acc", "Test Acc", "Precision", "Recall", "F1", "ROC-AUC"]
    (REPORTS_DIR / "sentiment_comparison.md").write_text(
        "# IMDb sentiment: model comparison\n\n" + markdown_table(rows, cols, bold_row=best) + "\n"
    )

    bar_comparison({r["Model"]: r["Test Acc"] for r in rows}, best,
                   "IMDb sentiment: test accuracy by model", "Accuracy (10K held-out reviews)",
                   FIGURES_DIR / "sentiment_model_comparison.png", xlim=(0.8, 0.93))
    confusion_matrices({best: confusion_matrix(y_test, preds[best])}, LABELS,
                       FIGURES_DIR / "sentiment_confusion_matrix.png")


if __name__ == "__main__":
    np.random.seed(SEED)
    main()
