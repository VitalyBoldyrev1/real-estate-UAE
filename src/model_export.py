import mlflow
import mlflow.catboost
from mlflow.tracking import MlflowClient


class ModelExporter:
    def __init__(
            self,
            experiment_name="CatBoost_Dubai_Real_Estate",
            model_run_name="dubai_catboost_v11",
            model_base_name="catboost_dubai_property_model"):
        self.experiment_name = experiment_name
        self.model_run_name = model_run_name
        self.model_base_name = model_base_name
        mlflow.set_tracking_uri(
            "file:///Users/vitalyboldyrev/real_estate_uae/mlruns")
        self.client = MlflowClient()

    def get_run_id_by_run_name(self):
        experiment = self.client.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            raise ValueError(f"Experiment '{self.experiment_name}' not found.")

        runs = self.client.search_runs(
            [experiment.experiment_id],
            filter_string=f"tags.mlflow.runName = '{self.model_run_name}'",
            order_by=["attributes.start_time DESC"],
            max_results=1
        )

        if not runs:
            raise ValueError(
                f"No run found with name '{
                    self.model_run_name}'.")

        return runs[0].info.run_id

    def export_model(self, output_path="dubai_model_v11.cbm"):
        run_id = self.get_run_id_by_run_name()
        print(f"Found run_id: {run_id} for model name '{self.model_run_name}'")

        model_uri = f"runs:/{run_id}/{self.model_base_name}"
        model = mlflow.catboost.load_model(model_uri)

        model.save_model(output_path)
        print(f"Model exported to: {output_path}")
