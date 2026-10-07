# IMDb sentiment: model comparison

| Model | CV Acc | Test Acc | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| **Linear SVM** | 0.909 | 0.914 | 0.909 | 0.920 | 0.915 | 0.972 |
| Logistic Regression | 0.909 | 0.913 | 0.905 | 0.923 | 0.914 | 0.972 |
| Naive Bayes | 0.886 | 0.888 | 0.883 | 0.896 | 0.889 | 0.955 |
| Course baseline (TF-IDF 5k + LR) | 0.883 | 0.885 | 0.875 | 0.900 | 0.887 | 0.956 |
