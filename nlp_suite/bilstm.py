"""BiLSTM + attention emotion model (trained in notebooks/02_emotion_bilstm_attention.ipynb).

Needs TensorFlow: pip install -r requirements-dl.txt
"""

import json
import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from nlp_suite import MODELS_DIR
from nlp_suite.text import clean_for_bilstm

BILSTM_DIR = MODELS_DIR / "emotion_bilstm"
MAX_LEN = 50


class AttentionLayer(layers.Layer):
    """Additive self-attention (same definition as in the training notebook)."""

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super().build(input_shape)

    def scores(self, x):
        e = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)  # (batch, time, 1)
        return tf.nn.softmax(e, axis=1)

    def call(self, x):
        return tf.reduce_sum(x * self.scores(x), axis=1)


class BiLSTMEmotion:
    def __init__(self, model_dir: Path = BILSTM_DIR):
        model_dir = Path(model_dir)
        self.model = tf.keras.models.load_model(
            model_dir / "emotion_bilstm_attention.keras",
            custom_objects={"AttentionLayer": AttentionLayer},
            compile=False,
        )
        with open(model_dir / "tokenizer_emotion.pkl", "rb") as f:
            self.tokenizer = pickle.load(f)
        self.labels = json.loads((model_dir / "label_classes.json").read_text())
        # Sub-model that exposes the BiLSTM states, used to recompute attention weights
        self._states = tf.keras.Model(self.model.input, self.model.get_layer("bilstm").output)

    def _encode(self, texts):
        cleaned = [clean_for_bilstm(t) for t in texts]
        seqs = self.tokenizer.texts_to_sequences(cleaned)
        return tf.keras.utils.pad_sequences(seqs, maxlen=MAX_LEN, padding="post", truncating="post"), cleaned

    def predict_proba_batch(self, texts) -> np.ndarray:
        x, _ = self._encode(texts)
        return self.model.predict(x, verbose=0, batch_size=256)

    def predict_proba(self, text: str) -> dict[str, float]:
        probs = self.predict_proba_batch([text])[0]
        return {self.labels[i]: float(p) for i, p in enumerate(probs)}

    def word_importance(self, text: str, label: str) -> list[tuple[str, float]]:
        """Occlusion importance: drop in P(label) when each word is removed.

        Positive = the word supports `label`; negative = it argues against it.
        Used instead of the attention weights, which are nearly uniform for this
        model (see `attention`).
        """
        words = clean_for_bilstm(text).split()[:MAX_LEN]
        if not words:
            return []
        k = self.labels.index(label)
        variants = [" ".join(words)] + [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
        # Variants are already cleaned; cleaning is idempotent so _encode is safe to reuse
        probs = self.predict_proba_batch(variants)[:, k]
        return list(zip(words, (probs[0] - probs[1:]).tolist()))

    def attention(self, text: str) -> list[tuple[str, float]]:
        """Attention weight for each word the model actually saw (after cleaning).

        Diagnostic only: the tanh scores saturate near 1 for every real word,
        and padded positions are not masked, so the weights over words come
        out almost uniform.
        """
        x, cleaned = self._encode([text])
        words = cleaned[0].split()[:MAX_LEN]
        if not words:
            return []
        att = self.model.get_layer("attention").scores(self._states(x))
        w = att.numpy()[0, : len(words), 0]
        w = w / w.sum()
        return list(zip(words, w.tolist()))
