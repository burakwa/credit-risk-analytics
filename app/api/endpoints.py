from fastapi import APIRouter, HTTPException
from ml.inference import predictor
from app.api.schemas.predict import CreditRiskFeatures, PredictionInput, PredictionOutput, ProbabilityOutput
from typing import List

router = APIRouter(prefix="/api/v1", tags=["predictions"])

FEATURE_MAPPING = {
    "age": "Age",
    "sex": "Sex",
    "job": "Job",
    "housing": "Housing",
    "saving_accounts": "Saving accounts",
    "checking_account": "Checking account",
    "credit_amount": "Credit amount",
    "duration": "Duration",
    "purpose": "Purpose"
}


def map_features(features: List[CreditRiskFeatures]) -> List[dict]:
    return [
        {FEATURE_MAPPING.get(k, k): v for k, v in f.model_dump().items()}
        for f in features
    ]


@router.post("/predict", response_model=PredictionOutput, summary="Kredi Riski Tahmini")
def predict(input: PredictionInput):
    """
    Verilen özellikler için kredi riski tahmini yapar.
    - **0**: Düşük risk (bad)
    - **1**: Yüksek risk (good)
    """
    try:
        features = map_features(input.features)
        predictions_dict = predictor.predict(features)
        return PredictionOutput(predictions=predictions_dict["predictions"])
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Model yüklenemedi: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict-proba", response_model=ProbabilityOutput, summary="Kredi Riski Olasılık Tahmini")
def predict_proba(input: PredictionInput):
    """
    Verilen özellikler için kredi riski olasılıklarını döndürür.
    Her kayıt için [P(bad), P(good)] formatında döner.
    """
    try:
        features = map_features(input.features)
        probabilities_dict = predictor.predict_proba(features)
        return ProbabilityOutput(probabilities=probabilities_dict["probabilities"])
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Model yüklenemedi: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

