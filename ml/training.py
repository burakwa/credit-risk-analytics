import logging
from typing import Any
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib
import os


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPipeline:
    def __init__(self , file_path : str , target_column : str , test_size : float = 0.2 , random_state : int = 42):
        self.file_path = file_path
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state


    def load_and_preprocess(self):
        try:
            df = pd.read_csv(self.file_path)
            print("Data loaded successfully")
            df = (df
                .drop(columns=['Unnamed: 0'], errors='ignore')
                .fillna({'Saving accounts': 'unknown', 'Checking account': 'unknown'})
                )
            return df
        except Exception as e:
            print("Error in loading data",e)


    def split_data(self,df):
        try:
            X = df.drop(self.target_column, axis=1)
            y = df[self.target_column].map({'good': 1, 'bad': 0})
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
            print("Data split successfully")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            print("Error in splitting data",e)
    
class ModelTrainer:

    def __init__(self , model : Any = None):
        self.model = model if model is not None else XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            scale_pos_weight = 0.30,
            subsample = 1,
            colsample_bytree = 0.9,
            learning_rate = 0.2,
            max_depth = 4,
            n_estimators = 100,
            random_state=42)

    def _build_pipeline(self) -> Pipeline:
        binary_cols = ['Sex']
        categorical_cols = ['Housing', 'Saving accounts', 'Checking account', 'Purpose']
        numerical_cols = ['Age', 'Credit amount', 'Duration']
        categorical_transformer = Pipeline(steps=[
            ('imputer' , SimpleImputer(strategy='constant',fill_value='unknown')),
            ('onehot', OneHotEncoder(drop='if_binary', sparse_output=False,handle_unknown='ignore'))
            ])
        
        binary_transformer = Pipeline(steps=[
            ('binary_encoder', OrdinalEncoder(
                handle_unknown='use_encoded_value',
                unknown_value=-1
            ))
        ])

        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols),
                ('bin', binary_transformer, binary_cols)
            ])
        return Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', self.model)
        ])

    def train_model(self , X_train , y_train):
        try:
            self.pipeline = self._build_pipeline()
            self.pipeline.fit(X_train, y_train)
            print("Model trained successfully")
            return self.pipeline
        except Exception as e:
            print("Error in training model",e)

    def save_pipeline(self , path : str):
        try:
            if not hasattr(self, 'pipeline'):
                logger.error("Pipeline not found. Run train_model() first.")
                return
            os.makedirs(os.path.dirname(path), exist_ok=True)
            joblib.dump(self.pipeline, path)
            logger.info("Pipeline saved successfully to %s", path)
        except Exception as e:
            logger.error("Error in saving pipeline: %s", e)

if __name__ == "__main__":
    DATA_PATH ="../data/raw/german_credit_data.csv"
    PIPELINE_OUTPUT_PATH = "../artifacts/model.pkl"
    
    TARGET_COLUMN = 'Risk'
    
    data_pipeline = DataPipeline(file_path=DATA_PATH, target_column=TARGET_COLUMN)
    df = data_pipeline.load_and_preprocess()

    if df is not None:
        split_result = data_pipeline.split_data(df)
        if split_result is not None:
            X_train, X_test, y_train, y_test = split_result
            model_trainer = ModelTrainer()
            model_trainer.train_model(X_train, y_train)
            model_trainer.save_pipeline(PIPELINE_OUTPUT_PATH)