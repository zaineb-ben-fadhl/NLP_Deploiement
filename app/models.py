from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional


# =========================
# REQUEST MODELS
# =========================

class TextRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    text: str = Field(
        ...,
        min_length=1,
        description="Texte à analyser (NLP)"
    )


class TextBatchRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    texts: List[str] = Field(
        ...,
        min_length=1,
        description="Liste de textes à analyser"
    )


# =========================
# RESPONSE MODELS
# =========================

class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    text: Optional[str] = Field(
        None,
        description="Texte analysé (optionnel pour batch)"
    )
    prediction: int = Field(
        ...,
        description="Classe prédite (0 = non-stress, 1 = stress)"
    )
    label: str = Field(
        ...,
        description="Label lisible (Non-Stress / Stress)"
    )
    probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probabilité associée à la classe 1 (stress)"
    )
    risk_level: str = Field(
        ...,
        description="Niveau de risque (Low / Medium / High)"
    )
    model_name: Optional[str] = Field(
        None,
        description="Nom du modèle utilisé"
    )


class BatchPredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    predictions: List[PredictionResponse]
    count: int


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool


# =========================
# DRIFT RESPONSE
# =========================

class DriftCheckResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    features_analyzed: int
    features_drifted: int
    drift_percentage: float
    risk_level: str
