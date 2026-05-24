from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from src.modeling import EncoderClassifier
from src.preprocessing import clean_text


@dataclass(frozen=True)
class Prediction:
    label: str
    probability_human: float
    probability_machine: float


class TextDetectionModel:
    """Transformer inference wrapper for binary text detection.

    Label order:
    - class 0: human-written
    - class 1: machine-generated
    """

    def __init__(
        self,
        model_name: str,
        checkpoint_path: str,
        device: str | None = None,
        max_length: int = 256,
        dropout: float = 0.2,
    ):
        self.model_name = model_name
        self.checkpoint_path = Path(checkpoint_path)
        self.max_length = max_length

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        state_dict, metadata = self._load_checkpoint(self.checkpoint_path)
        checkpoint_model_name = metadata.get("model_name")
        if checkpoint_model_name and checkpoint_model_name != model_name:
            raise ValueError(
                f"Checkpoint was trained with {checkpoint_model_name!r}, but MODEL_NAME is {model_name!r}"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = EncoderClassifier(model_name=model_name, dropout=dropout)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, text: str) -> Prediction:
        text = clean_text(text)

        encoded = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        logits = self.model(**encoded)
        probabilities = torch.softmax(logits, dim=-1)[0].cpu().tolist()

        probability_human, probability_machine = self._postprocess_probabilities(probabilities)
        label = "machine" if probability_machine >= 0.5 else "human"

        return Prediction(
            label=label,
            probability_human=probability_human,
            probability_machine=probability_machine,
        )

    @staticmethod
    def _postprocess_probabilities(probabilities: list[float]) -> tuple[float, float]:
        if len(probabilities) != 2:
            raise ValueError(f"Expected 2 logits for binary classification, got {len(probabilities)}")

        return float(probabilities[0]), float(probabilities[1])

    @staticmethod
    def _load_checkpoint(checkpoint_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

        metadata: dict[str, Any] = {}
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            metadata = {key: value for key, value in checkpoint.items() if key != "state_dict"}
            checkpoint = checkpoint["state_dict"]
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
        return checkpoint, metadata
