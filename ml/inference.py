from ml.training import logger
import logging
from pathlib import Path
from typing import List, Dict, Union, Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

class ModelPredictor:
    def __init__(self, model_path: Union[str, Path]):
        self.model_path = Path(model_path)
        self._pipeline: Pipeline = None

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None
    def load(self) -> None:
        if not self.is_loaded:
            try:
                self._pipeline = joblib.load(self.model_path)
                logger.info(f"Model loaded from {self.model_path}")
            except FileNotFoundError as e:
                logger.error(f"Model not found at {self.model_path}")
                raise FileNotFoundError(f"Model not found at {self.model_path}") from e
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                raise
    def _prepare_data(self, features: List[Dict[str, Any]]) -> pd.DataFrame:
        if not features:
            raise ValueError("Features list cannot be empty")   
        df = pd.DataFrame(features)    
        logger.info(f"Data prepared with {len(df)} records")
        return df
    def predict(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not self.is_loaded:
            self.load()
        df = self._prepare_data(features)
        prediction = self._pipeline.predict(df)
        logger.info(f"Prediction made with {len(df)} records")
        return {"predictions": prediction.tolist()}
        
    def predict_proba(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not self.is_loaded:
            self.load()
        if not hasattr(self._pipeline, "predict_proba"):
            raise AttributeError("Model does not support probability prediction")
        df = self._prepare_data(features)
        prediction = self._pipeline.predict_proba(df)
        logger.info(f"Prediction made with {len(df)} records")
        return {"probabilities": prediction.tolist()}

DEFAULT_MODEL_PATH = "../artifacts/model.pkl"
predictor = ModelPredictor(model_path=DEFAULT_MODEL_PATH)
