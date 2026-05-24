import torch
from torch import nn
from transformers import AutoModel


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    masked_hidden = last_hidden_state * mask
    token_count = mask.sum(dim=1).clamp(min=1e-6)
    return masked_hidden.sum(dim=1) / token_count


class EncoderClassifier(nn.Module):
    """Encoder + mean pooling + MLP head used by the training notebook."""

    def __init__(self, model_name: str, dropout: float = 0.2) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        mid_size = max(hidden_size // 2, 128)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, mid_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid_size, 2),
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        encoder_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**encoder_kwargs)
        pooled_embedding = mean_pool(outputs.last_hidden_state, attention_mask)
        pooled_embedding = pooled_embedding.to(self.classifier[0].weight.dtype)
        return self.classifier(pooled_embedding)
