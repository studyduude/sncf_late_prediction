import numpy as np
import pandas as pd
import xgboost as xgb
from preprocessing import load_and_process
from scipy.sparse import csr_matrix
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm


def perform_grid_search_cv(
    model: RegressorMixin | ClassifierMixin,
    X: csr_matrix,
    y: csr_matrix,
    param_grid: dict[str, list[float | int]],
    cv: int = 5,
) -> RegressorMixin | ClassifierMixin:
    grid_search = GridSearchCV(model, param_grid=param_grid, cv=cv)
    grid_search.fit(X, y)
    return grid_search.best_estimator_


def compare_models_with_grid_search_cv(
    x_train: csr_matrix,
    y_train: csr_matrix,
    x_test: csr_matrix,
    y_test: csr_matrix,
    model_params: list[dict],
    models: list[RegressorMixin | ClassifierMixin],
    model_names: list[str],
    save_path: str = "../data/grid_search_results.csv",
) -> pd.DataFrame:
    """Perform grid search with cross validation on different models and return the results in a DataFrame."""
    results = []
    for model, model_name, param_grid in tqdm(zip(models, model_names, model_params)):
        best_model = perform_grid_search_cv(model, x_train, y_train, param_grid)
        score = best_model.score(x_test, y_test)  # R^2
        rmse = np.sqrt(mean_squared_error(y_true=y_test, y_pred=best_model.predict(x_test)))
        results.append(
            {
                "Model": model_name,
                "R^2": score,
                "RMSE": rmse,
                "Best Parameters": best_model.get_params(),
            }
        )

    # Save the results
    results_df = pd.DataFrame(results)
    results_df.to_csv(save_path, index=False)
    return results_df
