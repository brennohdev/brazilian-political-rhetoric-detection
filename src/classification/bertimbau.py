from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from src.classification.base import BaseClassifier
from src.schemas.prediction import Prediction
from src.schemas.speech import SpeechSegment
from src.taxonomy.registry import TaxonomyRegistry


class SegmentDataset(Dataset):
    """PyTorch dataset for speech segments."""

    def __init__(
        self,
        texts: list[str],
        labels: list[list[int]] | None,
        tokenizer,
        max_length: int = 512,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item


class BERTimbauModel(nn.Module):
    """BERTimbau with sigmoid classification head."""

    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled = outputs.last_hidden_state[:, 0, :]  # CLS token
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


class BERTimbauClassifier(BaseClassifier):
    """Fine-tuned BERTimbau for segment-level multi-label classification.

    Each segment is classified independently for each of the 6 techniques.
    The decision threshold is tuned on the development set.
    """

    MODEL_NAME = "neuralmind/bert-base-portuguese-cased"

    def __init__(
        self,
        taxonomy: TaxonomyRegistry,
        checkpoint_path: Path | None = None,
        threshold: float = 0.5,
    ) -> None:
        self._taxonomy = taxonomy
        self._technique_names = taxonomy.technique_names
        self._num_labels = len(self._technique_names)
        self._threshold = threshold
        self._device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self._model = BERTimbauModel(self.MODEL_NAME, self._num_labels)

        if checkpoint_path and checkpoint_path.exists():
            state = torch.load(checkpoint_path, map_location=self._device)
            self._model.load_state_dict(state)
            logger.info(f"Loaded checkpoint from {checkpoint_path}")

        self._model.to(self._device)
        self._model.eval()

    @property
    def model_id(self) -> str:
        return "bertimbau"

    def classify(self, segment: SpeechSegment) -> list[Prediction]:
        """Classify a single segment."""
        self._model.eval()
        encoding = self._tokenizer(
            segment.text,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )
        encoding = {k: v.to(self._device) for k, v in encoding.items()}

        with torch.no_grad():
            logits = self._model(**encoding)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        predictions = []
        for i, (prob, name) in enumerate(zip(probs, self._technique_names)):
            if prob >= self._threshold:
                predictions.append(
                    Prediction(
                        technique=name,
                        span=segment.text,  # Full segment (no span-level detection)
                        start_offset=0,
                        end_offset=len(segment.text),
                        confidence=float(prob),
                        model_id=self.model_id,
                        segment_id=segment.segment_id,
                    )
                )
        return predictions

    def train_model(
        self,
        train_texts: list[str],
        train_labels: list[list[int]],
        dev_texts: list[str],
        dev_labels: list[list[int]],
        epochs: int = 10,
        lr: float = 2e-5,
        batch_size: int = 16,
        output_dir: Path = Path("models/bertimbau"),
    ) -> dict[str, list[float]]:
        """Fine-tune the model on labeled data.

        Args:
            train_texts: Training segment texts.
            train_labels: Binary label vectors (one per technique).
            dev_texts: Development set texts.
            dev_labels: Development set labels.
            epochs: Maximum training epochs.
            lr: Learning rate.
            batch_size: Training batch size.
            output_dir: Directory for saving checkpoints.

        Returns:
            Training history dict with loss and F1 per epoch.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        self._model.train()
        self._model.to(self._device)

        train_dataset = SegmentDataset(train_texts, train_labels, self._tokenizer)
        dev_dataset = SegmentDataset(dev_texts, dev_labels, self._tokenizer)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        dev_loader = DataLoader(dev_dataset, batch_size=batch_size)

        optimizer = torch.optim.AdamW(self._model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        best_f1 = 0.0
        history: dict[str, list[float]] = {"loss": [], "dev_f1": []}
        per_technique_history: dict[str, list[float]] = {
            t: [] for t in self._technique_names
        }

        for epoch in range(epochs):
            # Training
            self._model.train()
            total_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(self._device)
                attention_mask = batch["attention_mask"].to(self._device)
                labels = batch["labels"].to(self._device)

                logits = self._model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            # Evaluation (overall + per-technique)
            dev_f1, per_tech_f1 = self._evaluate_f1_detailed(dev_loader)
            history["loss"].append(avg_loss)
            history["dev_f1"].append(dev_f1)
            for tech_name, f1_val in per_tech_f1.items():
                per_technique_history[tech_name].append(f1_val)

            logger.info(
                f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, "
                f"dev_macro_f1={dev_f1:.4f}"
            )

            # Save best model
            if dev_f1 > best_f1:
                best_f1 = dev_f1
                checkpoint_path = output_dir / "best_model.pt"
                torch.save(self._model.state_dict(), checkpoint_path)
                logger.info(f"    New best model saved (F1={best_f1:.4f})")

        # Load best model
        best_path = output_dir / "best_model.pt"
        if best_path.exists():
            state = torch.load(best_path, map_location=self._device)
            self._model.load_state_dict(state)

        self._model.eval()

        # Save training charts
        history["per_technique_f1"] = per_technique_history
        self._save_training_charts(history, output_dir)

        return history

    def _save_training_charts(
        self, history: dict[str, list[float]], output_dir: Path
    ) -> None:
        """Generate and save training visualization charts."""
        try:
            import matplotlib.pyplot as plt

            figures_dir = output_dir / "figures"
            figures_dir.mkdir(parents=True, exist_ok=True)

            epochs = range(1, len(history["loss"]) + 1)

            # Chart 1: Loss + Dev F1 (dual axis)
            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Training Loss", color="tab:red")
            ax1.plot(epochs, history["loss"], "r-o", label="Train Loss", markersize=5)
            ax1.tick_params(axis="y", labelcolor="tab:red")

            ax2 = ax1.twinx()
            ax2.set_ylabel("Dev Macro-F1", color="tab:blue")
            ax2.plot(epochs, history["dev_f1"], "b-s", label="Dev Macro-F1", markersize=5)
            ax2.tick_params(axis="y", labelcolor="tab:blue")

            # Mark best epoch
            best_epoch = int(np.argmax(history["dev_f1"])) + 1
            best_f1 = max(history["dev_f1"])
            ax2.axvline(best_epoch, color="green", linestyle="--", alpha=0.5)
            ax2.annotate(
                f"Best: epoch {best_epoch}\nF1={best_f1:.4f}",
                xy=(best_epoch, best_f1),
                xytext=(best_epoch + 0.5, best_f1 - 0.05),
                fontsize=9,
                arrowprops=dict(arrowstyle="->", color="green"),
            )

            fig.suptitle("BERTimbau Training: Loss and Dev Macro-F1")
            fig.tight_layout()
            fig.savefig(figures_dir / "training_curve.png", dpi=300, bbox_inches="tight")
            fig.savefig(figures_dir / "training_curve.pdf", bbox_inches="tight")
            plt.close(fig)

            # Chart 2: Per-technique F1 evolution (if available)
            if "per_technique_f1" in history:
                fig, ax = plt.subplots(figsize=(12, 7))
                for technique, f1_values in history["per_technique_f1"].items():
                    ax.plot(epochs, f1_values, "-o", label=technique, markersize=4)
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Dev F1")
                ax.set_title("Per-Technique F1 Evolution During Training")
                ax.legend(loc="lower right")
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, 1)
                fig.tight_layout()
                fig.savefig(figures_dir / "per_technique_f1.png", dpi=300, bbox_inches="tight")
                fig.savefig(figures_dir / "per_technique_f1.pdf", bbox_inches="tight")
                plt.close(fig)

            logger.info(f"Training charts saved to {figures_dir}")
        except ImportError:
            logger.warning("matplotlib not available — skipping chart generation")

    def tune_threshold(
        self, dev_texts: list[str], dev_labels: list[list[int]]
    ) -> float:
        """Grid search threshold on dev set for optimal macro-F1."""
        dev_dataset = SegmentDataset(dev_texts, dev_labels, self._tokenizer)
        dev_loader = DataLoader(dev_dataset, batch_size=16)

        # Get all predictions
        all_probs = []
        all_labels = []
        self._model.eval()
        with torch.no_grad():
            for batch in dev_loader:
                input_ids = batch["input_ids"].to(self._device)
                attention_mask = batch["attention_mask"].to(self._device)
                logits = self._model(input_ids, attention_mask)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)
                all_labels.append(batch["labels"].numpy())

        all_probs = np.vstack(all_probs)
        all_labels = np.vstack(all_labels)

        best_threshold = 0.5
        best_f1 = 0.0

        for threshold in np.arange(0.05, 0.55, 0.05):
            preds = (all_probs >= threshold).astype(int)
            f1 = self._compute_macro_f1(all_labels, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        self._threshold = best_threshold
        logger.info(f"Optimal threshold: {best_threshold:.2f} (macro-F1={best_f1:.4f})")
        return best_threshold

    def _evaluate_f1(self, loader: DataLoader) -> float:
        """Compute macro-F1 on a data loader."""
        all_probs = []
        all_labels = []
        self._model.eval()
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self._device)
                attention_mask = batch["attention_mask"].to(self._device)
                logits = self._model(input_ids, attention_mask)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)
                all_labels.append(batch["labels"].numpy())

        all_probs = np.vstack(all_probs)
        all_labels = np.vstack(all_labels)
        preds = (all_probs >= self._threshold).astype(int)
        return self._compute_macro_f1(all_labels, preds)

    def _evaluate_f1_detailed(
        self, loader: DataLoader
    ) -> tuple[float, dict[str, float]]:
        """Compute macro-F1 and per-technique F1 on a data loader."""
        all_probs = []
        all_labels = []
        self._model.eval()
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self._device)
                attention_mask = batch["attention_mask"].to(self._device)
                logits = self._model(input_ids, attention_mask)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)
                all_labels.append(batch["labels"].numpy())

        all_probs = np.vstack(all_probs)
        all_labels = np.vstack(all_labels)
        preds = (all_probs >= self._threshold).astype(int)

        per_technique: dict[str, float] = {}
        f1s = []
        for i, name in enumerate(self._technique_names):
            tp = ((preds[:, i] == 1) & (all_labels[:, i] == 1)).sum()
            fp = ((preds[:, i] == 1) & (all_labels[:, i] == 0)).sum()
            fn = ((preds[:, i] == 0) & (all_labels[:, i] == 1)).sum()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            per_technique[name] = float(f1)
            f1s.append(f1)

        macro_f1 = float(np.mean(f1s))
        return macro_f1, per_technique

    @staticmethod
    def _compute_macro_f1(labels: np.ndarray, preds: np.ndarray) -> float:
        """Compute macro-F1 across all labels."""
        f1s = []
        for i in range(labels.shape[1]):
            tp = ((preds[:, i] == 1) & (labels[:, i] == 1)).sum()
            fp = ((preds[:, i] == 1) & (labels[:, i] == 0)).sum()
            fn = ((preds[:, i] == 0) & (labels[:, i] == 1)).sum()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            f1s.append(f1)
        return float(np.mean(f1s))
