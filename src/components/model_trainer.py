import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from src.exception import CustomException
from src.logger import get_logger
from src.utils import evaluate_models, save_object


logger = get_logger(__name__)


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join(
        "artifacts", "model.pkl"
    )


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):

        try:
            logger.info("Splitting train and test input data")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "LinearRegression": LinearRegression(),

                "DecisionTree": DecisionTreeRegressor(
                    random_state=42
                ),

                "RandomForest": RandomForestRegressor(
                    random_state=42,
                    n_estimators=100
                ),

                "KNeighbors": KNeighborsRegressor(
                    n_neighbors=5
                )
            }

            model_report: dict = evaluate_models(
                X_train,
                y_train,
                X_test,
                y_test,
                models
            )

            best_model_score = max(model_report.values())

            best_model_name = max(
                model_report,
                key=model_report.get
            )

            best_model = models[best_model_name]

            logger.info(
                f"Best Model: {best_model_name} "
                f"with R2 Score: {best_model_score:.4f}"
            )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logger.info(
                f"Best Model saved successfully: {best_model_name}"
            )

            return best_model_name, best_model_score

        except Exception as e:
            raise CustomException(e, sys)