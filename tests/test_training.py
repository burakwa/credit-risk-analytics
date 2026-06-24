"""
ml/training.py — DataPipeline ve ModelTrainer testleri.
"""
import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from sklearn.pipeline import Pipeline

from ml.training import DataPipeline, ModelTrainer


# ===========================================================================
# DataPipeline — __init__
# ===========================================================================
class TestDataPipelineInit:
    def test_default_values(self):
        dp = DataPipeline(file_path="dummy.csv", target_column="Risk")
        assert dp.file_path == "dummy.csv"
        assert dp.target_column == "Risk"
        assert dp.test_size == 0.2
        assert dp.random_state == 42

    def test_custom_values(self):
        dp = DataPipeline(
            file_path="other.csv",
            target_column="Target",
            test_size=0.3,
            random_state=0,
        )
        assert dp.test_size == 0.3
        assert dp.random_state == 0


# ===========================================================================
# DataPipeline — load_and_preprocess
# ===========================================================================
class TestLoadAndPreprocess:
    def test_loads_csv_successfully(self, sample_csv):
        dp = DataPipeline(file_path=sample_csv, target_column="Risk")
        df = dp.load_and_preprocess()
        assert df is not None
        assert isinstance(df, pd.DataFrame)

    def test_returns_correct_row_count(self, sample_csv, sample_df):
        dp = DataPipeline(file_path=sample_csv, target_column="Risk")
        df = dp.load_and_preprocess()
        assert len(df) == len(sample_df)

    def test_removes_unnamed_column(self, sample_csv_with_unnamed):
        """index=True ile oluşturulan CSV'deki 'Unnamed: 0' kaldırılmalı."""
        dp = DataPipeline(file_path=sample_csv_with_unnamed, target_column="Risk")
        df = dp.load_and_preprocess()
        assert "Unnamed: 0" not in df.columns

    def test_fills_saving_accounts_na(self, sample_csv):
        dp = DataPipeline(file_path=sample_csv, target_column="Risk")
        df = dp.load_and_preprocess()
        assert df["Saving accounts"].isna().sum() == 0
        assert (df["Saving accounts"] == "unknown").any()

    def test_fills_checking_account_na(self, sample_csv):
        dp = DataPipeline(file_path=sample_csv, target_column="Risk")
        df = dp.load_and_preprocess()
        assert df["Checking account"].isna().sum() == 0
        assert (df["Checking account"] == "unknown").any()

    def test_returns_none_on_bad_path(self):
        dp = DataPipeline(file_path="nonexistent_file.csv", target_column="Risk")
        result = dp.load_and_preprocess()
        assert result is None

    def test_logs_error_on_bad_path(self, caplog):
        import logging
        dp = DataPipeline(file_path="nonexistent_file.csv", target_column="Risk")
        with caplog.at_level(logging.ERROR, logger="ml.training"):
            dp.load_and_preprocess()
        assert any("Error" in r.message for r in caplog.records)


# ===========================================================================
# DataPipeline — split_data
# ===========================================================================
class TestSplitData:
    def test_returns_four_splits(self, sample_csv):
        dp = DataPipeline(file_path=sample_csv, target_column="Risk")
        df = dp.load_and_preprocess()
        result = dp.split_data(df)
        assert result is not None
        assert len(result) == 4

    def test_split_sizes_are_correct(self, sample_csv, sample_df):
        dp = DataPipeline(file_path=sample_csv, target_column="Risk", test_size=0.4)
        df = dp.load_and_preprocess()
        X_train, X_test, y_train, y_test = dp.split_data(df)
        total = len(sample_df)
        # train + test = total
        assert len(X_train) + len(X_test) == total

    def test_target_column_not_in_features(self, sample_csv):
        dp = DataPipeline(file_path=sample_csv, target_column="Risk")
        df = dp.load_and_preprocess()
        X_train, X_test, y_train, y_test = dp.split_data(df)
        assert "Risk" not in X_train.columns
        assert "Risk" not in X_test.columns

    def test_labels_are_binary(self, sample_csv):
        dp = DataPipeline(file_path=sample_csv, target_column="Risk")
        df = dp.load_and_preprocess()
        X_train, X_test, y_train, y_test = dp.split_data(df)
        all_labels = pd.concat([y_train, y_test])
        assert set(all_labels.unique()).issubset({0, 1})

    def test_good_maps_to_1_bad_maps_to_0(self, sample_csv, sample_df):
        dp = DataPipeline(file_path=sample_csv, target_column="Risk")
        df = dp.load_and_preprocess()
        X_train, X_test, y_train, y_test = dp.split_data(df)
        all_labels = pd.concat([y_train, y_test])
        original_risk = sample_df["Risk"]
        good_count = (original_risk == "good").sum()
        bad_count = (original_risk == "bad").sum()
        assert (all_labels == 1).sum() == good_count
        assert (all_labels == 0).sum() == bad_count

    def test_reproducible_with_same_random_state(self, sample_csv):
        dp = DataPipeline(file_path=sample_csv, target_column="Risk", random_state=7)
        df = dp.load_and_preprocess()
        result1 = dp.split_data(df)
        result2 = dp.split_data(df)
        pd.testing.assert_frame_equal(result1[0].reset_index(drop=True),
                                      result2[0].reset_index(drop=True))

    def test_returns_none_on_missing_column(self, sample_df):
        """Hedef sütun yoksa hata loglanmalı, None dönmeli."""
        dp = DataPipeline(file_path="dummy.csv", target_column="NonExistent")
        result = dp.split_data(sample_df)
        assert result is None


# ===========================================================================
# ModelTrainer — __init__ ve _build_pipeline
# ===========================================================================
class TestModelTrainerInit:
    def test_default_model_is_xgbclassifier(self):
        from xgboost import XGBClassifier
        trainer = ModelTrainer()
        assert isinstance(trainer.model, XGBClassifier)

    def test_custom_model_is_used(self):
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression()
        trainer = ModelTrainer(model=lr)
        assert trainer.model is lr


class TestBuildPipeline:
    def test_returns_sklearn_pipeline(self):
        trainer = ModelTrainer()
        pipe = trainer._build_pipeline()
        assert isinstance(pipe, Pipeline)

    def test_pipeline_has_preprocessor_and_classifier(self):
        trainer = ModelTrainer()
        pipe = trainer._build_pipeline()
        step_names = [name for name, _ in pipe.steps]
        assert "preprocessor" in step_names
        assert "classifier" in step_names


# ===========================================================================
# ModelTrainer — train_model
# ===========================================================================
class TestTrainModel:
    def test_train_model_returns_pipeline(self, sample_csv):
        dp = DataPipeline(file_path=sample_csv, target_column="Risk")
        df = dp.load_and_preprocess()
        X_train, X_test, y_train, y_test = dp.split_data(df)

        # Job sütununu çıkar (pipeline tanımlı değil)
        X_train = X_train.drop(columns=["Job"], errors="ignore")
        X_test = X_test.drop(columns=["Job"], errors="ignore")

        from sklearn.linear_model import LogisticRegression
        trainer = ModelTrainer(model=LogisticRegression(max_iter=200))
        result = trainer.train_model(X_train, y_train)
        assert isinstance(result, Pipeline)

    def test_pipeline_attribute_set_after_training(self, sample_csv):
        dp = DataPipeline(file_path=sample_csv, target_column="Risk")
        df = dp.load_and_preprocess()
        X_train, _, y_train, _ = dp.split_data(df)
        X_train = X_train.drop(columns=["Job"], errors="ignore")

        from sklearn.linear_model import LogisticRegression
        trainer = ModelTrainer(model=LogisticRegression(max_iter=200))
        trainer.train_model(X_train, y_train)
        assert hasattr(trainer, "pipeline")

    def test_trained_pipeline_can_predict(self, sample_csv):
        dp = DataPipeline(file_path=sample_csv, target_column="Risk")
        df = dp.load_and_preprocess()
        X_train, X_test, y_train, y_test = dp.split_data(df)
        X_train = X_train.drop(columns=["Job"], errors="ignore")
        X_test = X_test.drop(columns=["Job"], errors="ignore")

        from sklearn.linear_model import LogisticRegression
        trainer = ModelTrainer(model=LogisticRegression(max_iter=200))
        trainer.train_model(X_train, y_train)
        preds = trainer.pipeline.predict(X_test)
        assert len(preds) == len(X_test)
        assert set(preds).issubset({0, 1})


# ===========================================================================
# ModelTrainer — save_pipeline
# ===========================================================================
class TestSavePipeline:
    def test_saves_file_to_disk(self, tmp_path, sample_csv):
        dp = DataPipeline(file_path=sample_csv, target_column="Risk")
        df = dp.load_and_preprocess()
        X_train, _, y_train, _ = dp.split_data(df)
        X_train = X_train.drop(columns=["Job"], errors="ignore")

        from sklearn.linear_model import LogisticRegression
        trainer = ModelTrainer(model=LogisticRegression(max_iter=200))
        trainer.train_model(X_train, y_train)

        save_path = str(tmp_path / "artifacts" / "model.pkl")
        trainer.save_pipeline(save_path)
        assert os.path.exists(save_path)

    def test_saved_file_is_loadable(self, tmp_path, sample_csv):
        import joblib
        dp = DataPipeline(file_path=sample_csv, target_column="Risk")
        df = dp.load_and_preprocess()
        X_train, _, y_train, _ = dp.split_data(df)
        X_train = X_train.drop(columns=["Job"], errors="ignore")

        from sklearn.linear_model import LogisticRegression
        trainer = ModelTrainer(model=LogisticRegression(max_iter=200))
        trainer.train_model(X_train, y_train)
        save_path = str(tmp_path / "model.pkl")
        trainer.save_pipeline(save_path)

        loaded = joblib.load(save_path)
        assert isinstance(loaded, Pipeline)

    def test_save_without_training_logs_error(self, tmp_path, caplog):
        import logging
        trainer = ModelTrainer()
        save_path = str(tmp_path / "model.pkl")
        with caplog.at_level(logging.ERROR, logger="ml.training"):
            trainer.save_pipeline(save_path)
        assert any("Pipeline not found" in r.message for r in caplog.records)
        assert not os.path.exists(save_path)
