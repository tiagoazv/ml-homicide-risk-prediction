"""Modelos finais para previsão suavizada e classificação de alto risco."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .prepare_data import (
    FEATURE_COLUMNS_BASE,
    FEATURE_COLUMNS_FINAL,
    SMOOTHED_RATE_LOG_TARGET_COLUMN,
    SMOOTHED_RATE_PERCENT_TARGET_COLUMN,
    SMOOTHED_RATE_TARGET_COLUMN,
    smoothed_target_columns,
    split_by_year,
)


class SmoothedMeanRateRegressor(BaseEstimator, RegressorMixin):
    """Baseline que repete a média da taxa nos três anos históricos."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SmoothedMeanRateRegressor":
        if "taxa_media_3_anos" not in X.columns:
            raise ValueError("A feature taxa_media_3_anos é obrigatória.")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return 100 * np.log1p(X["taxa_media_3_anos"].to_numpy())


def expected_calibration_error(
    observed: pd.Series, probability: np.ndarray, bins: int = 10
) -> float:
    """Calcula o ECE por faixas de probabilidade igualmente espaçadas."""

    if bins <= 0:
        raise ValueError("bins deve ser positivo.")
    actual = np.asarray(observed, dtype=float)
    predicted = np.asarray(probability, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError("observed e probability devem ter o mesmo tamanho.")
    edges = np.linspace(0, 1, bins + 1)
    error = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (predicted >= lower) & (
            predicted <= upper if index == bins - 1 else predicted < upper
        )
        if mask.any():
            error += mask.mean() * abs(actual[mask].mean() - predicted[mask].mean())
    return float(error)


def calibrate_validation_probabilities(
    validation_probability: np.ndarray,
    validation_observed: pd.Series,
    probability: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Aplica calibração isotônica ajustada somente na validação."""

    if validation_observed.nunique() < 2:
        return np.asarray(probability, dtype=float), False
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(validation_probability, validation_observed.to_numpy())
    return calibrator.predict(np.asarray(probability, dtype=float)), True


def build_smoothed_rate_models(
    random_state: int = 42,
    horizon: int = 3,
    model_params: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Retorna baselines e modelos para o horizonte informado."""

    params = model_params or {}
    ridge_params = dict(params.get("ridge", {}))
    random_forest_params = dict(params.get("random_forest", {}))
    gradient_params = dict(params.get("hist_gradient_boosting", {}))
    random_forest_params.setdefault("random_state", random_state)
    random_forest_params.setdefault("n_jobs", -1)
    gradient_params.setdefault("random_state", random_state)

    return {
        f"baseline_media_{horizon}_anos": SmoothedMeanRateRegressor(),
        "ridge": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=ridge_params.pop("alpha", 1.0), **ridge_params)),
            ]
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=random_forest_params.pop("n_estimators", 200),
            min_samples_leaf=random_forest_params.pop("min_samples_leaf", 10),
            **random_forest_params,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=gradient_params.pop("max_iter", 200),
            learning_rate=gradient_params.pop("learning_rate", 0.05),
            max_leaf_nodes=gradient_params.pop("max_leaf_nodes", 15),
            l2_regularization=gradient_params.pop("l2_regularization", 1.0),
            **gradient_params,
        ),
    }


def _authorized_features(
    data: pd.DataFrame, feature_columns: list[str] | None
) -> list[str]:
    authorized_columns = set(FEATURE_COLUMNS_BASE + FEATURE_COLUMNS_FINAL)
    features = list(feature_columns or [c for c in FEATURE_COLUMNS_BASE if c in data])
    if not features:
        raise ValueError("Nenhuma feature autorizada disponível.")
    unauthorized = set(features).difference(authorized_columns)
    missing = set(features).difference(data.columns)
    if unauthorized:
        raise ValueError(f"Feature não autorizada: {sorted(unauthorized)}")
    if missing:
        raise ValueError(f"Features ausentes: {sorted(missing)}")
    return features


def evaluate_smoothed_rate_predictions(
    actual_log_rate: pd.Series,
    predicted_log_rate: np.ndarray,
    future_rates: pd.Series,
    current_rates: pd.Series,
) -> dict[str, float]:
    """Calcula métricas na escala log, de taxa e de variação percentual."""

    predicted_rates = np.maximum(0, np.expm1(predicted_log_rate / 100))
    actual_rates = future_rates.to_numpy()
    current = current_rates.to_numpy()
    metrics = {
        "mae_log_100": float(mean_absolute_error(actual_log_rate, predicted_log_rate)),
        "rmse_log_100": float(
            np.sqrt(mean_squared_error(actual_log_rate, predicted_log_rate))
        ),
        "medae_log_100": float(
            median_absolute_error(actual_log_rate, predicted_log_rate)
        ),
        "r2_log_100": float(r2_score(actual_log_rate, predicted_log_rate)),
        "mae_taxa": float(mean_absolute_error(actual_rates, predicted_rates)),
        "medae_taxa": float(median_absolute_error(actual_rates, predicted_rates)),
    }
    valid = current > 0
    if valid.any():
        actual_percent = 100 * (actual_rates[valid] - current[valid]) / current[valid]
        predicted_percent = 100 * (
            predicted_rates[valid] - current[valid]
        ) / current[valid]
        metrics["mae_percentual"] = float(
            mean_absolute_error(actual_percent, predicted_percent)
        )
        metrics["medae_percentual"] = float(
            median_absolute_error(actual_percent, predicted_percent)
        )
    return metrics


def run_smoothed_rate_experiment(
    data: pd.DataFrame,
    train_end: int = 2014,
    validation_end: int = 2016,
    random_state: int = 42,
    model_output_dir: str | Path | None = None,
    feature_columns: list[str] | None = None,
    horizon: int = 3,
    include_ensemble: bool = False,
    model_params: dict[str, dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Treina o experimento para a média das taxas do horizonte futuro."""

    target_column, log_target_column, percent_target_column = smoothed_target_columns(
        horizon
    )

    required = {
        target_column,
        log_target_column,
        percent_target_column,
        "taxa_media_3_anos",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Colunas do alvo suavizado ausentes: {sorted(missing)}")

    splits = split_by_year(data, train_end=train_end, validation_end=validation_end)
    models = build_smoothed_rate_models(
        random_state=random_state,
        horizon=horizon,
        model_params=model_params,
    )
    features = _authorized_features(data, feature_columns)
    if model_output_dir is not None:
        Path(model_output_dir).mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    predictions_by_split: dict[str, dict[str, np.ndarray]] = {
        split_name: {} for split_name in splits
    }

    def append_results(
        model_name: str,
        split_name: str,
        split: pd.DataFrame,
        predicted: np.ndarray,
    ) -> None:
        metric = {"modelo": model_name, "divisao": split_name, "n": len(split)}
        metric.update(
            evaluate_smoothed_rate_predictions(
                split[log_target_column],
                predicted,
                split[target_column],
                split["taxa_media_3_anos"],
            )
        )
        metric_rows.append(metric)

        predicted_rates = np.maximum(0, np.expm1(predicted / 100))
        current = split["taxa_media_3_anos"].to_numpy()
        predicted_percent = np.full(current.shape, np.nan, dtype=float)
        np.divide(
            100 * (predicted_rates - current),
            current,
            out=predicted_percent,
            where=current > 0,
        )
        prediction_rows.append(
            pd.DataFrame(
                {
                    "municipio_codigo": split["municipio_codigo"].to_numpy(),
                    "municipio": split["municipio"].to_numpy(),
                    "ano": split["ano"].to_numpy(),
                    "ano_futuro_inicio": split["ano_futuro_inicio"].to_numpy(),
                    "ano_futuro": split["ano_futuro"].to_numpy(),
                    "modelo": model_name,
                    "divisao": split_name,
                    "populacao": split["populacao"].to_numpy()
                    if "populacao" in split
                    else np.nan,
                    "porte_populacional": split["porte_populacional"].astype("string").to_numpy()
                    if "porte_populacional" in split
                    else "indisponivel",
                    f"alvo_log_taxa_media_{horizon}_anos_100": split[
                        log_target_column
                    ].to_numpy(),
                    f"previsao_log_taxa_media_{horizon}_anos_100": predicted,
                    f"alvo_taxa_media_{horizon}_anos": split[
                        target_column
                    ].to_numpy(),
                    "taxa_media_3_anos_atual": current,
                    f"taxa_media_{horizon}_anos_prevista": predicted_rates,
                    f"alvo_variacao_media_{horizon}_anos_percentual": split[
                        percent_target_column
                    ].to_numpy(),
                    f"previsao_variacao_media_{horizon}_anos_percentual": predicted_percent,
                }
            )
        )

    for model_name, model in models.items():
        model.fit(
            splits["train"][features],
            splits["train"][log_target_column],
        )
        if model_output_dir is not None:
            joblib.dump(model, Path(model_output_dir) / f"{model_name}.joblib")

        for split_name, split in splits.items():
            predicted = model.predict(split[features])
            predictions_by_split[split_name][model_name] = predicted
            append_results(model_name, split_name, split, predicted)

    if include_ensemble:
        ensemble_name = "ensemble_media"
        member_names = [
            f"baseline_media_{horizon}_anos",
            "random_forest",
            "hist_gradient_boosting",
        ]
        for split_name, split in splits.items():
            missing_members = set(member_names).difference(
                predictions_by_split[split_name]
            )
            if missing_members:
                raise ValueError(
                    f"Membros ausentes no ensemble: {sorted(missing_members)}"
                )
            predicted = np.mean(
                [predictions_by_split[split_name][name] for name in member_names],
                axis=0,
            )
            append_results(ensemble_name, split_name, split, predicted)

    return pd.DataFrame(metric_rows), pd.concat(prediction_rows, ignore_index=True)


def build_risk_classifiers(random_state: int = 42) -> dict[str, Any]:
    """Classificadores para identificar o quartil superior da taxa futura."""

    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=random_state,
        ),
    }


def run_risk_classification_experiment(
    data: pd.DataFrame,
    train_end: int = 2014,
    validation_end: int = 2016,
    random_state: int = 42,
    model_output_dir: str | Path | None = None,
    feature_columns: list[str] | None = None,
    risk_quantile: float = 0.75,
    horizon: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Classifica cidades no quartil superior do alvo futuro.

    O ponto de corte é calculado somente no treino de cada janela. Isso evita
    usar a distribuição dos anos de validação ou teste para definir o rótulo.
    """

    if not 0 < risk_quantile < 1:
        raise ValueError("risk_quantile deve estar entre zero e um.")
    target_column, _, _ = smoothed_target_columns(horizon)
    if target_column not in data:
        raise ValueError("O alvo de taxa suavizada é obrigatório.")

    splits = split_by_year(data, train_end=train_end, validation_end=validation_end)
    features = _authorized_features(data, feature_columns)
    threshold = float(
        splits["train"][target_column].quantile(risk_quantile)
    )
    train_target = splits["train"][target_column].ge(threshold).astype(int)
    if train_target.nunique() < 2:
        raise ValueError("O corte de risco produziu somente uma classe no treino.")

    models = build_risk_classifiers(random_state=random_state)
    if model_output_dir is not None:
        Path(model_output_dir).mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    for model_name, model in models.items():
        model.fit(splits["train"][features], train_target)
        if model_output_dir is not None:
            joblib.dump(model, Path(model_output_dir) / f"{model_name}.joblib")

        validation = splits["validation"]
        validation_probability = model.predict_proba(validation[features])[:, 1]
        validation_observed = validation[target_column].ge(threshold).astype(int)
        for split_name, split in splits.items():
            observed = split[target_column].ge(threshold).astype(int)
            probability = model.predict_proba(split[features])[:, 1]
            calibrated_probability, calibration_applied = calibrate_validation_probabilities(
                validation_probability,
                validation_observed,
                probability,
            )
            predicted = (probability >= 0.5).astype(int)
            calibrated_predicted = (calibrated_probability >= 0.5).astype(int)
            metric = {
                "modelo": model_name,
                "divisao": split_name,
                "n": len(split),
                "limiar_taxa_alto_risco": threshold,
                "calibracao_aplicada": calibration_applied,
                "proporcao_alto_risco_observado": float(observed.mean()),
                "balanced_accuracy": float(
                    balanced_accuracy_score(observed, predicted)
                ),
                "f1": float(f1_score(observed, predicted, zero_division=0)),
                "precision": float(
                    precision_score(observed, predicted, zero_division=0)
                ),
                "recall": float(recall_score(observed, predicted, zero_division=0)),
                "brier_score": float(brier_score_loss(observed, probability)),
                "ece_10": expected_calibration_error(observed, probability),
                "log_loss": float(log_loss(observed, probability, labels=[0, 1]))
                if observed.nunique() > 1
                else np.nan,
                "balanced_accuracy_calibrado": float(
                    balanced_accuracy_score(observed, calibrated_predicted)
                ),
                "f1_calibrado": float(
                    f1_score(observed, calibrated_predicted, zero_division=0)
                ),
                "brier_score_calibrado": float(
                    brier_score_loss(observed, calibrated_probability)
                ),
                "ece_10_calibrado": expected_calibration_error(
                    observed, calibrated_probability
                ),
                "log_loss_calibrado": float(
                    log_loss(observed, calibrated_probability, labels=[0, 1])
                )
                if observed.nunique() > 1
                else np.nan,
                "roc_auc_calibrado": float(
                    roc_auc_score(observed, calibrated_probability)
                )
                if observed.nunique() > 1
                else np.nan,
                "roc_auc": float(roc_auc_score(observed, probability))
                if observed.nunique() > 1
                else np.nan,
            }
            metric_rows.append(metric)
            prediction_rows.append(
                pd.DataFrame(
                    {
                        "municipio_codigo": split["municipio_codigo"].to_numpy(),
                        "municipio": split["municipio"].to_numpy(),
                        "ano": split["ano"].to_numpy(),
                        "ano_futuro_inicio": split["ano_futuro_inicio"].to_numpy(),
                        "ano_futuro": split["ano_futuro"].to_numpy(),
                        "modelo": model_name,
                        "divisao": split_name,
                        "populacao": split["populacao"].to_numpy()
                        if "populacao" in split
                        else np.nan,
                        "limiar_taxa_alto_risco": threshold,
                        f"taxa_futura_media_{horizon}_anos": split[
                            target_column
                        ].to_numpy(),
                        "risco_alto_observado": observed.to_numpy(),
                        "probabilidade_alto_risco": probability,
                        "probabilidade_alto_risco_calibrada": calibrated_probability,
                        "risco_alto_previsto": predicted,
                        "risco_alto_previsto_calibrado": calibrated_predicted,
                    }
                )
            )

    return pd.DataFrame(metric_rows), pd.concat(prediction_rows, ignore_index=True), threshold
