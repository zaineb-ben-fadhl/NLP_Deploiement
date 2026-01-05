from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
import traceback
from typing import Optional
import asyncio
import time

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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
# MODEL LOADING CONFIGURATION
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = os.getenv("MODEL_DIR", "artifacts/models/deberta_dreaddit_best")
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "128"))
MODEL_LOAD_TIMEOUT = int(os.getenv("MODEL_LOAD_TIMEOUT", "300"))

tokenizer: Optional[AutoTokenizer] = None
model: Optional[AutoModelForSequenceClassification] = None
model_load_error: Optional[str] = None
model_load_time: Optional[float] = None


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def _softmax(logits: np.ndarray) -> np.ndarray:
    """Compute softmax probabilities"""
    exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp / np.sum(exp, axis=-1, keepdims=True)


def _risk_level(probability_class_1: float) -> str:
    """Determine risk level based on stress probability"""
    if probability_class_1 < 0.3:
        return "Low"
    if probability_class_1 < 0.7:
        return "Medium"
    return "High"


def _predict_one(text: str):
    """Make prediction on a single text"""
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

    proba_1 = float(probs[1]) if probs.shape[0] == 2 else float(np.max(probs))

    label = "Stress" if pred == 1 else "Non-Stress"
    risk = _risk_level(proba_1)

    return pred, label, proba_1, risk


# ============================================================
# MODEL LOADING WITH DETAILED LOGGING
# ============================================================

async def _load_model_async():
    """Load model asynchronously with detailed logging"""
    global tokenizer, model, model_load_error, model_load_time
    
    overall_start = time.time()
    
    try:
        logger.info("="*60)
        logger.info("🔄 Starting model load process")
        logger.info(f"📂 Model directory: {MODEL_DIR}")
        logger.info(f"🖥️  Device: {DEVICE}")
        logger.info(f"⏱️  Timeout: {MODEL_LOAD_TIMEOUT}s")
        logger.info("="*60)
        
        # Check if directory exists
        if not os.path.exists(MODEL_DIR):
            raise FileNotFoundError(f"❌ Model directory not found: {MODEL_DIR}")
        
        # List files and calculate size
        files = os.listdir(MODEL_DIR)
        logger.info(f"📁 Files found ({len(files)}): {files}")
        
        total_size = 0
        for f in files:
            path = os.path.join(MODEL_DIR, f)
            if os.path.isfile(path):
                size = os.path.getsize(path)
                total_size += size
                logger.info(f"   - {f}: {size / (1024**2):.2f} MB")
        
        logger.info(f"💾 Total model size: {total_size / (1024**3):.2f} GB")
        logger.info("-"*60)
        
        # Load tokenizer
        logger.info("🔤 Loading tokenizer...")
        start_time = time.time()
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_DIR, 
            use_fast=False,
            local_files_only=True
        )
        tokenizer_time = time.time() - start_time
        logger.info(f"✅ Tokenizer loaded in {tokenizer_time:.2f}s")
        logger.info("-"*60)
        
        # Load model
        logger.info("🧠 Loading model (this may take 1-3 minutes for large models)...")
        start_time = time.time()
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_DIR,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        model_time = time.time() - start_time
        logger.info(f"✅ Model loaded in {model_time:.2f}s")
        logger.info("-"*60)
        
        # Move to device
        logger.info(f"📍 Moving model to {DEVICE}...")
        start_time = time.time()
        model.to(DEVICE)
        model.eval()
        device_time = time.time() - start_time
        logger.info(f"✅ Model moved to {DEVICE} in {device_time:.2f}s")
        logger.info("-"*60)
        
        model_load_time = time.time() - overall_start
        
        logger.info("="*60)
        logger.info("🎉 MODEL SUCCESSFULLY LOADED AND READY!")
        logger.info(f"⏱️  Total loading time: {model_load_time:.2f}s")
        logger.info("="*60)
        
        logger.info("model_loaded", extra={
            "custom_dimensions": {
                "event_type": "model_load",
                "model_dir": MODEL_DIR,
                "device": DEVICE,
                "status": "success",
                "model_size_gb": round(total_size / (1024**3), 2),
                "load_time_seconds": round(model_load_time, 2),
                "tokenizer_time": round(tokenizer_time, 2),
                "model_time": round(model_time, 2),
                "device_time": round(device_time, 2)
            }
        })
        
    except Exception as e:
        error_msg = f"❌ Model load failed: {str(e)}"
        logger.error("="*60)
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        logger.error("="*60)
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
        model_load_error = error_msg


@app.on_event("startup")
async def load_model():
    """Startup event with timeout and comprehensive logging"""
    global model_load_error
    
    logger.info("🚀 Application startup initiated")
    
    try:
        await asyncio.wait_for(
            _load_model_async(),
            timeout=MODEL_LOAD_TIMEOUT
        )
        logger.info("✅ Application startup completed successfully")
        
    except asyncio.TimeoutError:
        error_msg = f"⏱️ Model loading timed out after {MODEL_LOAD_TIMEOUT} seconds"
        logger.error(error_msg)
        model_load_error = error_msg
        
    except Exception as e:
        error_msg = f"💥 Unexpected error during startup: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        model_load_error = error_msg


# ============================================================
# GENERAL ENDPOINTS
# ============================================================

@app.get("/", tags=["General"])
def root():
    """Root endpoint with system information"""
    return {
        "message": "Dreaddit DeBERTa Prediction API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "device": DEVICE,
        "model_dir": MODEL_DIR,
        "model_loaded": model is not None,
        "model_load_time_seconds": round(model_load_time, 2) if model_load_time else None,
        "model_load_error": model_load_error,
    }


@app.get("/liveness", tags=["General"])
def liveness():
    """Liveness probe - always returns 200 if app is running"""
    return {
        "status": "alive",
        "timestamp": time.time()
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health():
    """Health check - returns 503 if model not loaded"""
    if model is None or tokenizer is None:
        detail = model_load_error if model_load_error else "Model not loaded"
        raise HTTPException(status_code=503, detail=detail)
    return {
        "status": "healthy",
        "model_loaded": True
    }


# ============================================================
# PREDICTION ENDPOINTS
# ============================================================

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(req: TextRequest):
    """Predict stress level for a single text"""
    if model is None or tokenizer is None:
        detail = model_load_error if model_load_error else "Model unavailable"
        raise HTTPException(status_code=503, detail=detail)

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
    """Predict stress level for multiple texts"""
    if model is None or tokenizer is None:
        detail = model_load_error if model_load_error else "Model unavailable"
        raise HTTPException(status_code=503, detail=detail)

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
    """Log drift detection results to Application Insights"""
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
    """Check for data drift between reference and production data"""
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
    """Manually trigger a drift alert"""
    logger.warning("manual_drift_alert", extra={
        "custom_dimensions": {
            "event_type": "manual_drift_alert",
            "alert_message": message,
            "severity": severity,
            "triggered_by": "api_endpoint",
        }
    })
    return {"status": "alert_sent"}