import sys
import os
import pandas as pd

from src.exception import CustomException
from src.utils import load_object
from src.logger import get_logger


logger = get_logger(__name__)


class CustomData:

    def __init__(
        self,
        age: int,
        sex: str,
        bmi: float,
        children: int,
        smoker: str,
        region: str
    ):
        self.age = age
        self.sex = sex
        self.bmi = bmi
        self.children = children
        self.smoker = smoker
        self.region = region

    def get_data_as_dataframe(self):

        try:
            custom_data_input_dict = {
                "age": [self.age],
                "sex": [self.sex],
                "bmi": [self.bmi],
                "children": [self.children],
                "smoker": [self.smoker],
                "region": [self.region]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)


class PredictPipeline:

    def __init__(self):

        self.model_path = os.path.join(
            "artifacts",
            "model.pkl"
        )

        self.preprocessor_path = os.path.join(
            "artifacts",
            "preprocessor.pkl"
        )

    def predict(self, features: pd.DataFrame):

        try:
            logger.info(
                "Loading Model and Preprocessor"
            )

            model = load_object(self.model_path)

            preprocessor = load_object(
                self.preprocessor_path
            )

            # Transform input data
            data_transformed = preprocessor.transform(
                features
            )

            # Make prediction
            prediction = model.predict(
                data_transformed
            )

            result = prediction[0]

            logger.info(
                f"Prediction completed: {result:.2f}"
            )

            return result

        except Exception as e:
            raise CustomException(e, sys)