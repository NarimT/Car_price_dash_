from mlflow.tracking import MlflowClient
client = MlflowClient()
client.get_registered_model("A3_st125983")