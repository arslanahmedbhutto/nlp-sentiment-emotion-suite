# Emotion classification: model comparison (test set, 2,000 texts)

| Model | Type | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|---|
| **Logistic Regression** | TF-IDF | 0.882 | 0.837 | 0.883 |
| BiLSTM + Attention | Deep learning | 0.876 | 0.834 | 0.878 |
| Linear SVM | TF-IDF | 0.884 | 0.829 | 0.884 |
| Naive Bayes | TF-IDF | 0.760 | 0.630 | 0.741 |

## F1 per emotion

| Model | anger | fear | joy | love | sadness | surprise |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.881 | 0.847 | 0.909 | 0.770 | 0.919 | 0.696 |
| BiLSTM + Attention | 0.880 | 0.853 | 0.898 | 0.789 | 0.911 | 0.671 |
| Linear SVM | 0.886 | 0.863 | 0.910 | 0.765 | 0.920 | 0.630 |
| Naive Bayes | 0.716 | 0.712 | 0.813 | 0.450 | 0.811 | 0.278 |
