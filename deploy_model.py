import mlflow
import pickle
import pandas as pd
from src import *
import joblib
# Load local model

local_path = "models/logistic_model.joblib"
predictor = joblib.load(local_path)
# Wrap model for MLflow
class CarPriceWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, predictor):
        self.predictor = predictor

    def predict(self, context, model_input: pd.DataFrame):
        return self.predictor.predict(model_input)

# Example input for MLflow schema
sample_data = [[2014, 103.52, 0 ,27, 21.14,]]
column_names = ['year', 'max_power', 'fuel','brand', 'mileage']
sample_df = pd.DataFrame(sample_data, columns=column_names)
sample_array = sample_df.to_numpy()
sample_with_bias = np.insert(sample_array, 0, 1, axis=1)

mlflow.set_tracking_uri("https://mlflow.ml.brain.cs.ait.ac.th/")
mlflow.set_experiment("A3_st125983")

with mlflow.start_run(run_name="logistic_regression_deploy") as run:
    mlflow.pyfunc.log_model(
        name="model",
        python_model=CarPriceWrapper(predictor),
        input_example=sample_with_bias
    )
    # Construct proper model URI to register
    model_uri = f"runs:/{run.info.run_id}/model"

# Register as a new version
registered_model = mlflow.register_model(
    model_uri=model_uri,
    name="A3_st125983"
)

print("Model deployed to MLflow!")