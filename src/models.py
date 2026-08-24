"""Modelos e métricas do MVP analítico."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .prepare_data import (
    FEATURE_COLUMNS,
    TARGET_LOG_COLUMN,
    TARGET_PERCENT_COLUMN,
    get_feature_columns,
    log_change_to_percent,
    split_by_year,
)


class LastValueRegressor(BaseEstimator, RegressorMixin):
    """Baseline que repete a última variação logarítmica observada."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LastValueRegressor":
        if "variacao_log_100_atual" not in X.columns:
            raise ValueError("A feature variacao_log_100_atual é obrigatória.")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["variacao_log_100_atual"].to_numpy()


def build_models(random_state: int = 42) -> dict[str, Any]:
    """Retorna o baseline e os dois modelos iniciais do projeto."""

    return {
        "baseline_zero": DummyRegressor(strategy="constant", constant=0.0),
        "baseline_last_value": LastValueRegressor(),
        "ridge": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=150,
            min_samples_leaf=10,
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def evaluate_predictions(
    actual: pd.Series, predicted: np.ndarray, raw_actual: pd.Series | None = None
) -> dict[str, float]:
    """Calcula métricas no alvo logarítmico e, quando possível, em percentual."""

    metrics = {
        "mae_log_100": float(mean_absolute_error(actual, predicted)),
        "rmse_log_100": float(
            np.sqrt(mean_squared_error(actual, predicted))
        ),
        "medae_log_100": float(median_absolute_error(actual, predicted)),
        "r2_log_100": float(r2_score(actual, predicted)),
    }

    if raw_actual is not None:
        valid = raw_actual.notna().to_numpy()
        if valid.any():
            actual_percent = raw_actual.to_numpy()[valid]
            predicted_percent = log_change_to_percent(predicted[valid])
            metrics["mae_percentual"] = float(
                mean_absolute_error(actual_percent, predicted_percent)
            )
            metrics["medae_percentual"] = float(
                median_absolute_error(actual_percent, predicted_percent)
            )
    return metrics


def evaluate_by_population_size(
    predictions: pd.DataFrame, split_name: str = "test"
) -> pd.DataFrame:
    """Calcula o MAE por porte populacional no período escolhido."""

    required = {"porte_populacional", "alvo_log_100", "previsao_log_100"}
    if not required.issubset(predictions.columns):
        raise ValueError("As previsões precisam conter o porte populacional.")

    rows = []
    selected = predictions[predictions["divisao"].eq(split_name)]
    for (model_name, size), group in selected.groupby(
        ["modelo", "porte_populacional"], observed=True
    ):
        rows.append(
            {
                "modelo": model_name,
                "divisao": split_name,
                "porte_populacional": size,
                "n": len(group),
                "mae_log_100": mean_absolute_error(
                    group["alvo_log_100"], group["previsao_log_100"]
                ),
                "rmse_log_100": np.sqrt(
                    mean_squared_error(
                        group["alvo_log_100"], group["previsao_log_100"]
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def run_experiment(
    data: pd.DataFrame,
    train_end: int = 2016,
    validation_end: int = 2019,
    random_state: int = 42,
    model_output_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Treina os modelos e retorna métricas e previsões dos três períodos."""

    splits = split_by_year(data, train_end=train_end, validation_end=validation_end)
    models = build_models(random_state=random_state)
    feature_columns = get_feature_columns(data)
    if model_output_dir is not None:
        Path(model_output_dir).mkdir(parents=True, exist_ok=True)
    metrics_rows: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []

    for model_name, model in models.items():
        model.fit(
            splits["train"][feature_columns], splits["train"][TARGET_LOG_COLUMN]
        )
        if model_output_dir is not None:
            joblib.dump(model, Path(model_output_dir) / f"{model_name}.joblib")

        for split_name, split in splits.items():
            predicted = model.predict(split[feature_columns])
            metric_row = {
                "modelo": model_name,
                "divisao": split_name,
                "n": len(split),
            }
            metric_row.update(
                evaluate_predictions(
                    split[TARGET_LOG_COLUMN],
                    predicted,
                    split[TARGET_PERCENT_COLUMN],
                )
            )
            metrics_rows.append(metric_row)

            predictions.append(
                pd.DataFrame(
                    {
                        "municipio_codigo": split["municipio_codigo"].to_numpy(),
                        "municipio": split["municipio"].to_numpy(),
                        "ano": split["ano"].to_numpy(),
                        "ano_futuro": split["ano_futuro"].to_numpy(),
                        "modelo": model_name,
                        "divisao": split_name,
                        "populacao": split["populacao"].to_numpy()
                        if "populacao" in split
                        else np.nan,
                        "porte_populacional": split["porte_populacional"].astype("string").to_numpy()
                        if "porte_populacional" in split
                        else "indisponivel",
                        "populacao_observada": split["populacao_observada"].to_numpy()
                        if "populacao_observada" in split
                        else False,
                        "alvo_log_100": split[TARGET_LOG_COLUMN].to_numpy(),
                        "previsao_log_100": predicted,
                        "alvo_percentual": split[TARGET_PERCENT_COLUMN].to_numpy(),
                        "previsao_percentual_estabilizada": log_change_to_percent(
                            predicted
                        ),
                    }
                )
            )

    return pd.DataFrame(metrics_rows), pd.concat(predictions, ignore_index=True)
