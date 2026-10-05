import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.exception import CustomException
from src.logger import get_logger
from src.utils import save_object


logger = get_logger(__name__)


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join(
        "artifacts", "preprocessor.pkl"
    )


class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = ["age", "bmi", "children"]
            categorical_columns = ["sex", "smoker", "region"]

            # Numerical pipeline
            numerical_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="median")
                    )
                ]
            )

            # Categorical pipeline
            categorical_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent")
                    ),
                    (
                        "onehot",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False
                        )
                    )
                ]
            )

            # Column transformer
            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "numerical_pipeline",
                        numerical_pipeline,
                        numerical_columns
                    ),
                    (
                        "categorical_pipeline",
                        categorical_pipeline,
                        categorical_columns
                    )
                ]
            )

            logger.info("Preprocessor object created successfully")

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(
        self,
        train_path: str,
        test_path: str
    ):
        try:
            # Read train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logger.info(
                "Train and test data loaded for transformation"
            )

            # Target column
            target_column = "charges"

            # Separate input and target features
            input_feature_train_df = train_df.drop(
                columns=[target_column]
            )
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(
                columns=[target_column]
            )
            target_feature_test_df = test_df[target_column]

            # Get preprocessing object
            preprocessor_obj = self.get_data_transformer_object()

            logger.info(
                "Applying preprocessing on train and test data"
            )

            # Fit on training data and transform both datasets
            input_feature_train_arr = preprocessor_obj.fit_transform(
                input_feature_train_df
            )

            input_feature_test_arr = preprocessor_obj.transform(
                input_feature_test_df
            )

            # Combine features and target
            train_arr = np.c_[
                input_feature_train_arr,
                np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr,
                np.array(target_feature_test_df)
            ]

            # Save preprocessor
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            logger.info(
                "Preprocessor object saved successfully"
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":

    obj = DataTransformation()

    obj.initiate_data_transformation(
        "artifacts/train.csv",
        "artifacts/test.csv"
    )