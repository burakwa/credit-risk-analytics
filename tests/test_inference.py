"""
ml/inference.py — ModelPredictor testleri.
"""
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch
import joblib

from ml.inference import ModelPredictor


# ---------------------------------------------------------------------------
# Örnek feature listesi (pipeline'ın beklediği sütunlar)
# ---------------------------------------------------------------------------
VALID_FEATURES = [
    {
        "Age": 30, "Sex": "male", "Housing": "own",
        "Saving accounts": "little", "Checking account": "moderate",
        "Credit amount": 3000, "Duration": 24, "Purpose": "car",
    }
]

MULTI_FEATURES = VALID_FEATURES + [
    {
        "Age": 45, "Sex": "female", "Housing": "free",
        "Saving accounts": "rich", "Checking account": "little",
        "Credit amount": 8000, "Duration": 36, "Purpose": "education",
    }
]


# ===========================================================================
# ModelPredictor — __init__ ve is_loaded property
# ===========================================================================
class TestModelPredictorInit:
    def test_model_path_stored_as_path(self, tmp_path):
        p = tmp_path / "model.pkl"
        mp = ModelPredictor(model_path=str(p))
        assert isinstance(mp.model_path, Path)
        assert mp.model_path == p

    def test_is_loaded_false_initially(self, saved_pipeline_path):
        mp = ModelPredictor(model_path=saved_pipeline_path)
        assert mp.is_loaded is False

    def test_is_loaded_true_after_load(self, saved_pipeline_path):
        mp = ModelPredictor(model_path=saved_pipeline_path)
        mp.load()
        assert mp.is_loaded is True


# ===========================================================================
# ModelPredictor — load
# ===========================================================================
class TestModelPredictorLoad:
    def test_load_succeeds_with_valid_path(self, saved_pipeline_path):
        mp = ModelPredictor(model_path=saved_pipeline_path)
        mp.load()  # exception yükseltmemeli
        assert mp.is_loaded

    def test_load_raises_file_not_found(self, tmp_path):
        mp = ModelPredictor(model_path=str(tmp_path / "nonexistent.pkl"))
        with pytest.raises(FileNotFoundError):
            mp.load()

    def test_load_called_only_once(self, saved_pipeline_path):
        """İkinci çağrıda model tekrar yüklenmemeli (is_loaded guard)."""
        mp = ModelPredictor(model_path=saved_pipeline_path)
        mp.load()
        pipeline_ref = mp._pipeline
        mp.load()  # ikinci çağrı
        assert mp._pipeline is pipeline_ref   # aynı obje

    def test_load_logs_success(self, saved_pipeline_path, caplog):
        import logging
        mp = ModelPredictor(model_path=saved_pipeline_path)
        with caplog.at_level(logging.INFO, logger="ml.inference"):
            mp.load()
        assert any("loaded" in r.message.lower() for r in caplog.records)

    def test_load_logs_error_on_missing_file(self, tmp_path, caplog):
        import logging
        mp = ModelPredictor(model_path=str(tmp_path / "missing.pkl"))
        with caplog.at_level(logging.ERROR, logger="ml.inference"):
            with pytest.raises(FileNotFoundError):
                mp.load()
        assert any("not found" in r.message.lower() for r in caplog.records)


# ===========================================================================
# ModelPredictor — _prepare_data
# ===========================================================================
class TestPrepareData:
    def test_returns_dataframe(self, saved_pipeline_path):
        mp = ModelPredictor(model_path=saved_pipeline_path)
        df = mp._prepare_data(VALID_FEATURES)
        assert isinstance(df, pd.DataFrame)

    def test_row_count_matches_input(self, saved_pipeline_path):
        mp = ModelPredictor(model_path=saved_pipeline_path)
        df = mp._prepare_data(MULTI_FEATURES)
        assert len(df) == len(MULTI_FEATURES)

    def test_raises_on_empty_features(self, saved_pipeline_path):
        mp = ModelPredictor(model_path=saved_pipeline_path)
        with pytest.raises(ValueError, match="cannot be empty"):
            mp._prepare_data([])

    def test_columns_match_feature_keys(self, saved_pipeline_path):
        mp = ModelPredictor(model_path=saved_pipeline_path)
        df = mp._prepare_data(VALID_FEATURES)
        for key in VALID_FEATURES[0]:
            assert key in df.columns


# ===========================================================================
# ModelPredictor — predict
# ===========================================================================
class TestPredict:
    def test_predict_returns_dict_with_predictions_key(self, saved_pipeline_path):
        mp = ModelPredictor(model_path=saved_pipeline_path)
        result = mp.predict(VALID_FEATURES)
        assert "predictions" in result

    def test_predict_output_is_list(self, saved_pipeline_path):
        mp = ModelPredictor(model_path=saved_pipeline_path)
        result = mp.predict(VALID_FEATURES)
        assert isinstance(result["predictions"], list)

    def test_predict_output_length_matches_input(self, saved_pipeline_path):
        mp = ModelPredictor(model_path=saved_pipeline_path)
        result = mp.predict(MULTI_FEATURES)
        assert len(result["predictions"]) == len(MULTI_FEATURES)

    def test_predict_values_are_binary(self, saved_pipeline_path):
        mp = ModelPredictor(model_path=saved_pipeline_path)
        result = mp.predict(MULTI_FEATURES)
        for val in result["predictions"]:
            assert val in (0, 1)

    def test_predict_auto_loads_model(self, saved_pipeline_path):
        """predict() çağrısı önce load() tetiklemeli."""
        mp = ModelPredictor(model_path=saved_pipeline_path)
        assert not mp.is_loaded
        mp.predict(VALID_FEATURES)
        assert mp.is_loaded

    def test_predict_raises_on_empty_features(self, saved_pipeline_path):
        mp = ModelPredictor(model_path=saved_pipeline_path)
        with pytest.raises(ValueError):
            mp.predict([])


# ===========================================================================
# ModelPredictor — predict_proba
# ===========================================================================
class TestPredictProba:
    def test_predict_proba_returns_dict_with_probabilities_key(self, saved_pipeline_path):
        mp = ModelPredictor(model_path=saved_pipeline_path)
        result = mp.predict_proba(VALID_FEATURES)
        assert "probabilities" in result

    def test_predict_proba_output_is_list_of_lists(self, saved_pipeline_path):
        mp = ModelPredictor(model_path=saved_pipeline_path)
        result = mp.predict_proba(VALID_FEATURES)
        proba = result["probabilities"]
        assert isinstance(proba, list)
        assert isinstance(proba[0], list)

    def test_predict_proba_each_row_sums_to_one(self, saved_pipeline_path):
        mp = ModelPredictor(model_path=saved_pipeline_path)
        result = mp.predict_proba(MULTI_FEATURES)
        for row in result["probabilities"]:
            assert abs(sum(row) - 1.0) < 1e-6

    def test_predict_proba_output_length_matches_input(self, saved_pipeline_path):
        mp = ModelPredictor(model_path=saved_pipeline_path)
        result = mp.predict_proba(MULTI_FEATURES)
        assert len(result["probabilities"]) == len(MULTI_FEATURES)

    def test_predict_proba_auto_loads_model(self, saved_pipeline_path):
        mp = ModelPredictor(model_path=saved_pipeline_path)
        assert not mp.is_loaded
        mp.predict_proba(VALID_FEATURES)
        assert mp.is_loaded

    def test_predict_proba_raises_when_no_proba_support(self, tmp_path):
        """predict_proba desteklemeyen modelde AttributeError yükseltmeli."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        # predict_proba olmayan sahte bir pipeline
        mock_pipeline = MagicMock(spec=Pipeline)
        del mock_pipeline.predict_proba  # spec üzerinden kaldır

        mp = ModelPredictor(model_path=str(tmp_path / "dummy.pkl"))
        mp._pipeline = mock_pipeline  # doğrudan enjekte et

        with pytest.raises(AttributeError, match="probability"):
            mp.predict_proba(VALID_FEATURES)

    def test_predict_proba_raises_on_empty_features(self, saved_pipeline_path):
        mp = ModelPredictor(model_path=saved_pipeline_path)
        with pytest.raises(ValueError):
            mp.predict_proba([])
