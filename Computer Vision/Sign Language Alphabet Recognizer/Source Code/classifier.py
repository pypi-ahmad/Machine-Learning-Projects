"""Sign Language Alphabet Recognizer -- lightweight sklearn classifier."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


class SignClassifier:
    """Thin wrapper around a trained sklearn classifier.

    The classifier is stored as a pickle file containing a dict::

        {"model": <sklearn estimator>, "labels": ["A", "B", ...]}
    """

    def __init__(self) -> None:
        self._model = None
        self._labels: list[str] = []
        self._class_indices: list[int] = []

    @property
    def ready(self) -> bool:
        return self._model is not None

    @property
    def labels(self) -> list[str]:
        return list(self._labels)

    # -- persistence -----------------------------------------------------

    def load(self, path: str | Path) -> None:
        """Load a trained model from *path*."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Model not found: {p}")
        with open(p, "rb") as f:
            bundle = pickle.load(f)
        self._model = bundle["model"]
        self._labels = bundle["labels"]
        self._class_indices = [
            int(cls) for cls in bundle.get("classes", range(len(self._labels)))
        ]

    def save(self, path: str | Path) -> None:
        """Persist current model + label list."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(
                {
                    "model": self._model,
                    "labels": self._labels,
                    "classes": self._class_indices,
                },
                f,
            )

    # -- training --------------------------------------------------------

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        labels: list[str],
        *,
        max_iter: int = 500,
        hidden_layer_sizes: tuple[int, ...] = (128, 64),
    ) -> dict[str, float]:
        """Train an MLP classifier and return training metrics.

        Parameters
        ----------
        X : array of shape (n_samples, 42)
        y : integer labels of shape (n_samples,)
        labels : ordered list of class names
        """
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        self._labels = list(labels)
        self._model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPClassifier(
                        hidden_layer_sizes=hidden_layer_sizes,
                        max_iter=max_iter,
                        early_stopping=True,
                        validation_fraction=0.1,
                        random_state=42,
                    ),
                ),
            ]
        )
        self._model.fit(X, y)
        self._class_indices = [int(cls) for cls in self._model.named_steps["mlp"].classes_]
        train_acc = float(self._model.score(X, y))
        return {"train_accuracy": train_acc}

    # -- inference -------------------------------------------------------

    def predict(self, features: np.ndarray) -> tuple[str, float]:
        """Classify a single (42,) feature vector.

        Returns (label, confidence).
        """
        if not self.ready:
            raise RuntimeError("Model not loaded -- call load() or train() first")
        probs = self._model.predict_proba(features.reshape(1, -1))[0]
        pos = int(np.argmax(probs))
        label_idx = self._class_indices[pos]
        return self._labels[label_idx], float(probs[pos])

    def predict_batch(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Classify a batch. Returns (predicted_indices, probabilities)."""
        if not self.ready:
            raise RuntimeError("Model not loaded")
        probs = self._model.predict_proba(X)
        preds = self._model.predict(X).astype(np.int64)
        class_to_pos = {cls: pos for pos, cls in enumerate(self._class_indices)}
        confs = np.array(
            [probs[i, class_to_pos[int(pred)]] for i, pred in enumerate(preds)],
            dtype=np.float32,
        )
        return preds, confs
