from dataclasses import dataclass

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.preprocessing import clean_text


@dataclass(frozen=True)
class Prediction:
    label: str
    probability_human: float
    probability_machine: float


class TextDetectionModel:
    """Transformer inference wrapper for binary text detection.

    Assumed label order:
    - class 0: human-written
    - class 1: machine-generated

    Check the label mapping in the training notebook before publishing.
    If the training code used the opposite order, swap the probabilities in
    `_postprocess_probabilities`.
    """

    def __init__(self, model_path: str, device: str | None = None, max_length: int = 512):
        self.model_path = model_path
        self.max_length = max_length

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
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

        logits = self.model(**encoded).logits
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

        # Current expected mapping: 0 = human, 1 = machine.
        return float(probabilities[0]), float(probabilities[1])
