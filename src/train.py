import numpy as np
import pandas as pd
import mlflow
import mlflow.catboost
import optuna
from optuna.samplers import TPESampler
from catboost import CatBoostRegressor
from sklearn.model_selection import cross_validate, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import shap
import matplotlib.pyplot as plt
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import os


class ModelTrainer:
    '''
    CatBoost model trainer with MLflow tracking and Optuna hyperparameter optimization.
    '''
    
    def __init__(
        self,
        experiment_name='CatBoost_Dubai_Real_Estate',
        model_base_name='catboost_dubai_property_model',
        tracking_uri='file:///Users/vitalyboldyrev/real_estate_uae/mlruns'
    ):
        '''
        Initialize trainer with MLflow configuration
        '''

        self.experiment_name = experiment_name
        self.model_base_name = model_base_name
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.client = MlflowClient()
        print(f'MLflow experiment set to: {self.experiment_name}')

    def _objective(self, trial, X_train, y_train, cat_features, n_splits):
        '''
        Optuna objective function for hyperparameter optimization
        '''

        params = {
            'iterations': trial.suggest_categorical('iterations', [500, 1000]),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.06),
            'depth': trial.suggest_int('depth', 5, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 20.0, log=True),
            'border_count': trial.suggest_categorical('border_count', [64, 128]),
            'random_strength': trial.suggest_float(
                'random_strength', 1e-2, 5.0, log=True
            ),
            'loss_function': 'RMSE',
            'eval_metric': 'RMSE',
            'cat_features': cat_features,
            'verbose': 0,
            'early_stopping_rounds': 50,
            'random_state': 42,
        }
        model = CatBoostRegressor(
            **params, thread_count=-1, sampling_frequency='PerTree'
        )

        splitter = TimeSeriesSplit(n_splits=n_splits)
        cv_results = cross_validate(
            model,
            X_train,
            y_train,
            scoring='neg_root_mean_squared_error',
            cv=splitter,
            n_jobs=1,
        )
        rmse = -cv_results['test_score'].mean()
        return rmse

    def get_next_version(self):
        '''
        Generate next model version tag based on existing MLflow runs
        '''

        try:
            exp = self.client.get_experiment_by_name(self.experiment_name)
            if exp is None:
                return 'v1'
            runs = self.client.search_runs(
                [exp.experiment_id],
                filter_string=f'tags.mlflow.runName LIKE \'{self.model_base_name}_v%\'',
                run_view_type=ViewType.ACTIVE_ONLY,
                order_by=['attributes.start_time DESC'],
                max_results=1,
            )
            if not runs:
                return 'v1'
            latest = runs[0].data.tags['mlflow.runName'].split('_v')[-1]
            return f'v{int(latest) + 1}'
        except Exception as e:
            print(f'Version resolution failed: {e}')
            return 'v1'

    def train_and_log_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        cat_features: list[str],
        n_trials: int = 10,
        cv_splits_for_optuna: int = 3,
    ) -> tuple[CatBoostRegressor, np.ndarray]:
        '''
        Main training pipeline - optimize hyperparameters, train model and log to MLflow
        '''

        # Convert column names to indices for CatBoost
        if cat_features and isinstance(X_train, pd.DataFrame):
            cat_features = [X_train.columns.get_loc(col) for col in cat_features]

        print('Running Optuna search...')
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=42, multivariate=True, n_startup_trials=10),
            pruner=optuna.pruners.MedianPruner(),
        )

        study.optimize(
            lambda trial: self._objective(
                trial, X_train, y_train, cat_features, cv_splits_for_optuna
            ),
            n_trials=n_trials,
            n_jobs=1,
            timeout=3 * 60 * 60,  # 3 hours
        )

        best_params = study.best_params
        print(f'Best parameters: {best_params}')

        final_model = CatBoostRegressor(
            **best_params,
            cat_features=cat_features,
            random_state=42,
            verbose=100,
            task_type='CPU',
            devices='0:1',
        )

        print('Fitting final model on full training set...')
        final_model.fit(X_train, y_train, plot=False)

        model_version_tag = self.get_next_version()
        run_name = f'{self.model_base_name}_{model_version_tag}'

        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            print(f'MLflow Run ID: {run_id}')
            mlflow.log_params(best_params)
            mlflow.log_param('optuna_trials', len(study.trials))
            mlflow.log_param('cv_splits', cv_splits_for_optuna)

            # Calculate predictions
            y_pred_train = final_model.predict(X_train)
            y_pred_test = final_model.predict(X_test)

            # Metrics accounting for log transformation of prices
            train_rmse = np.sqrt(
                mean_squared_error(np.expm1(y_train), np.expm1(y_pred_train))
            )
            test_rmse = np.sqrt(mean_squared_error(y_test, np.expm1(y_pred_test)))
            test_mae = mean_absolute_error(y_test, np.expm1(y_pred_test))
            test_r2 = r2_score(y_test, np.expm1(y_pred_test))
            # train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            # test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            # test_mae = mean_absolute_error(y_test, y_pred_test)
            # test_r2 = r2_score(y_test, y_pred_test)

            mlflow.log_metrics(
                {
                    'best_cv_rmse': study.best_value,
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'test_mae': test_mae,
                    'test_r2': test_r2,
                }
            )

            mlflow.set_tag('model_name', self.model_base_name)
            mlflow.set_tag('version', model_version_tag)

            mlflow.catboost.log_model(
                cb_model=final_model,
                artifact_path=self.model_base_name,
                registered_model_name=self.model_base_name,
            )
            print(f'Model logged to MLflow: {self.model_base_name}')

            print('Calculating SHAP values...')
            explainer = shap.TreeExplainer(final_model)
            shap_values = explainer.shap_values(X_train)

            shap_output_dir = 'shap_outputs'
            if not os.path.exists(shap_output_dir):
                os.makedirs(shap_output_dir)

            shap_summary_path = os.path.join(
                shap_output_dir, f'{run_name}_shap_summary_bar.png'
            )

            plt.figure()
            shap.summary_plot(
                shap_values, X_train, plot_type='bar', show=False, max_display=25
            )
            plt.tight_layout()
            plt.savefig(shap_summary_path)
            plt.close()
            mlflow.log_artifact(shap_summary_path, artifact_path='shap_plots')
            print(f'SHAP plot saved: {shap_summary_path}')

        print(f'\n--- Training Summary ({run_name}) ---')
        print(f'Best CV RMSE: {study.best_value:.4f}')
        print(f'Train RMSE: {train_rmse:.4f}')
        print(f'Test RMSE: {test_rmse:.4f}')
        print(f'Test MAE: {test_mae:.4f}')
        print(f'Test R²: {test_r2:.4f}')

        return final_model, y_pred_test
