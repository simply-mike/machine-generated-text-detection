from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Input text for classification")


class PredictionResponse(BaseModel):
    label: str
    probability_human: float
    probability_machine: float
