import os
from functools import lru_cache

from fastapi import FastAPI

from app.schemas import PredictionRequest, PredictionResponse
from src.inference import TextDetectionModel


MODEL_PATH = os.getenv("MODEL_PATH", "models/mdeberta_processed_article")
DEVICE = os.getenv("DEVICE")
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "512"))

app = FastAPI(
    title="Machine-Generated Text Detection API",
    description="Inference service for classifying text as human-written or machine-generated.",
    version="1.0.0",
)


@lru_cache(maxsize=1)
def get_model() -> TextDetectionModel:
    return TextDetectionModel(model_path=MODEL_PATH, device=DEVICE, max_length=MAX_LENGTH)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    prediction = get_model().predict(request.text)
    return PredictionResponse(
        label=prediction.label,
        probability_human=prediction.probability_human,
        probability_machine=prediction.probability_machine,
    )
