import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException

from app.schemas import PredictionRequest, PredictionResponse
from src.inference import TextDetectionModel


MODEL_NAME = os.getenv("MODEL_NAME", "microsoft/mdeberta-v3-base")
CHECKPOINT_PATH = os.getenv(
    "CHECKPOINT_PATH",
    os.getenv("MODEL_PATH", "checkpoints/task7/article_style/mdeberta_processed_article/best_model.pt"),
)
DEVICE = os.getenv("DEVICE")
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "256"))
DROPOUT = float(os.getenv("DROPOUT", "0.2"))

app = FastAPI(
    title="Machine-Generated Text Detection API",
    description="Inference service for classifying text as human-written or machine-generated.",
    version="1.0.0",
)


@lru_cache(maxsize=1)
def get_model() -> TextDetectionModel:
    return TextDetectionModel(
        model_name=MODEL_NAME,
        checkpoint_path=CHECKPOINT_PATH,
        device=DEVICE,
        max_length=MAX_LENGTH,
        dropout=DROPOUT,
    )


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def readiness_check() -> dict[str, str]:
    try:
        model = get_model()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "status": "ready",
        "model_name": model.model_name,
        "checkpoint_path": str(model.checkpoint_path),
        "device": str(model.device),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    prediction = get_model().predict(request.text)
    return PredictionResponse(
        label=prediction.label,
        probability_human=prediction.probability_human,
        probability_machine=prediction.probability_machine,
    )
