from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
import traceback
from typing import Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from app.models import (
    TextRequest,
    TextBatchRequest,
    PredictionResponse,
    BatchPredictionResponse,
    HealthResponse,
    DriftCheckResponse,
)
from app.drift_detect import detect_drift


# ============================================================
# LOGGING & APPLICATION INSIGHTS (SAFE IMPORT)
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dreaddit-deberta-api")

APPINSIGHTS_CONN = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")

AzureLogHandler = None
try:
    from opencensus.ext.azure.log_exporter import AzureLogHandler  # type: ignore
except Exception:
    AzureLogHandler = None

if APPINSIGHTS_CONN and AzureLogHandler:
    handler = AzureLogHandler(connection_string=APPINSIGHTS_CONN)
    logger.addHandler(handler)
    logger.info("app_startup", extra={
        "custom_dimensions": {
            "event_type": "startup",
            "status": "application_insights_connected"
        }
    })
else:
    logger.warning("app_startup", extra={
        "custom_dimensions": {
            "event_type": "startup",
            "status": "application_insights_not_configured_or_package_missing"
        }
    })


# ============================================================
# FASTAPI INIT
# ============================================================

app = FastAPI(
    title="Dreaddit DeBERTa Prediction API",
    description="API de prédiction (stress) + monitoring drift pour Dreaddit",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# MODEL LOADING
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Default fallback si la variable d'environnement n'est pas définie
MODEL_DIR = os.getenv("MODEL_DIR", "artifacts/models/deberta_dreaddit_best")

MAX_LENGTH = int(os.getenv("MAX_LENGTH", "128"))

tokenizer: Optional[AutoTokenizer] = None
model: Optional[AutoModelForSequenceClassification] = None


def _softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp / np.sum(exp, axis=-1, keepdims=True)


def _risk_level(probability_class_1: float) -> str:
    if probability_class_1 < 0.3:
        return "Low"
    if probability_class_1 < 0.7:
        return "Medium"
    return "High"


def _predict_one(text: str):
    if tokenizer is None or model is None:
        raise RuntimeError("Model not loaded")

    encoded = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    encoded = {k: v.to(DEVICE) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits.detach().cpu().numpy()[0]

    probs = _softmax(logits)
    pred = int(np.argmax(probs))

    # proba de la classe 1 (stress)
    proba_1 = float(probs[1]) if probs.shape[0] == 2 else float(np.max(probs))

    label = "Stress" if pred == 1 else "Non-Stress"
    risk = _risk_level(proba_1)

    return pred, label, proba_1, risk


@app.on_event("startup")
async def load_model():
    global tokenizer, model
    try:
        # ✅ Important: use_fast=False to avoid DeBERTa tokenizer conversion bug
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        model.to(DEVICE)
        model.eval()

        logger.info("model_loaded", extra={
            "custom_dimensions": {
                "event_type": "model_load",
                "model_dir": MODEL_DIR,
                "device": DEVICE,
                "status": "success"
            }
        })
    except Exception as e:
        logger.error("model_load_failed", extra={
            "custom_dimensions": {
                "event_type": "model_load",
                "model_dir": MODEL_DIR,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        })
        tokenizer = None
        model = None


# ============================================================
# GENERAL ENDPOINTS
# ============================================================

@app.get("/", tags=["General"])
def root():
    return {
        "message": "Dreaddit DeBERTa Prediction API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "device": DEVICE,
        "model_dir": MODEL_DIR,
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health():
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}


# ============================================================
# PREDICTION ENDPOINTS
# ============================================================

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(req: TextRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model unavailable")

    try:
        pred, label, proba_1, risk = _predict_one(req.text)

        logger.info("prediction", extra={
            "custom_dimensions": {
                "event_type": "prediction",
                "endpoint": "/predict",
                "prediction": pred,
                "label": label,
                "probability": proba_1,
                "risk_level": risk
            }
        })

        return {
            "text": req.text,
            "prediction": pred,
            "label": label,
            "probability": round(proba_1, 6),
            "risk_level": risk,
            "model_name": MODEL_DIR,
        }

    except Exception as e:
        logger.error("prediction_error", extra={
            "custom_dimensions": {
                "event_type": "prediction_error",
                "endpoint": "/predict",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        })
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(req: TextBatchRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model unavailable")

    try:
        out = []
        for text in req.texts:
            pred, label, proba_1, risk = _predict_one(text)
            out.append({
                "text": text,
                "prediction": pred,
                "label": label,
                "probability": round(proba_1, 6),
                "risk_level": risk,
                "model_name": MODEL_DIR,
            })

        logger.info("batch_prediction", extra={
            "custom_dimensions": {
                "event_type": "batch_prediction",
                "endpoint": "/predict/batch",
                "count": len(out),
            }
        })

        return {"predictions": out, "count": len(out)}

    except Exception as e:
        logger.error("batch_prediction_error", extra={
            "custom_dimensions": {
                "event_type": "batch_prediction_error",
                "endpoint": "/predict/batch",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        })
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# DRIFT LOGGING TO APPLICATION INSIGHTS
# ============================================================

def log_drift_to_insights(drift_results: dict):
    total = len(drift_results)
    drifted = sum(1 for r in drift_results.values() if r.get("drift_detected"))
    percentage = round((drifted / total) * 100, 2) if total else 0

    risk = "LOW" if percentage < 20 else "MEDIUM" if percentage < 50 else "HIGH"

    logger.warning("drift_detection", extra={
        "custom_dimensions": {
            "event_type": "drift_detection",
            "drift_percentage": percentage,
            "risk_level": risk,
        }
    })

    for feature, details in drift_results.items():
        if details.get("drift_detected"):
            logger.warning("feature_drift", extra={
                "custom_dimensions": {
                    "event_type": "feature_drift",
                    "feature_name": feature,
                    "p_value": float(details.get("p_value", 0)),
                    "statistic": float(details.get("statistic", 0)),
                    "type": details.get("type", "unknown"),
                }
            })

    return percentage, risk, drifted, total


# ============================================================
# DRIFT ENDPOINTS
# ============================================================

@app.post("/drift/check", response_model=DriftCheckResponse, tags=["Drift"])
def check_drift(
    threshold: float = 0.05,
    reference_file: str = "data/drift_reference.csv",
    production_file: str = "data/drift_production.csv",
):
    try:
        results = detect_drift(
            reference_file=reference_file,
            production_file=production_file,
            threshold=threshold,
        )

        percentage, risk, drifted, total = log_drift_to_insights(results)

        return {
            "status": "success",
            "features_analyzed": total,
            "features_drifted": drifted,
            "drift_percentage": percentage,
            "risk_level": risk,
        }

    except Exception:
        tb = traceback.format_exc()
        logger.error("drift_error", extra={
            "custom_dimensions": {
                "event_type": "drift_error",
                "traceback": tb
            }
        })
        raise HTTPException(status_code=500, detail="Drift check failed")


@app.post("/drift/alert", tags=["Drift"])
def manual_drift_alert(
    message: str = "Manual drift alert triggered",
    severity: str = "warning",
):
    logger.warning("manual_drift_alert", extra={
        "custom_dimensions": {
            "event_type": "manual_drift_alert",
            "alert_message": message,
            "severity": severity,
            "triggered_by": "api_endpoint",
        }
    })
    return {"status": "alert_sent"}
