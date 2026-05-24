from typing import Literal

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Input text for classification")


class PredictionResponse(BaseModel):
    label: Literal["human", "machine"]
    probability_human: float = Field(..., ge=0.0, le=1.0)
    probability_machine: float = Field(..., ge=0.0, le=1.0)
