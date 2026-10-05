"""Análise de sensibilidade de contagem com exposição populacional."""

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, median_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .population import filter_by_population
from .prepare_data import FEATURE_COLUMNS_FINAL, split_by_year, smoothed_target_columns


COUNT_TARGET = "homicidios_proxy_futuro_2_anos"
EXPOSURE_COLUMN = "exposicao_populacional_2_anos"
COUNT_FEATURES = [column for column in FEATURE_COLUMNS_FINAL if column != "log_populacao"]
ALPHA_GRID = (0.01, 0.1, 1.0)


class MeanExposureCountRegressor(BaseEstimator, RegressorMixin):
    """Baseline que repete a taxa histórica média e aplica a exposição."""

    def __init__(
        self,
        rate_column: str = "taxa_media_3_anos",
        exposure_column: str = EXPOSURE_COLUMN,
    ):
        self.rate_column = rate_column
        self.exposure_column = exposure_column

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MeanExposureCountRegressor":
        required = {self.rate_column, self.exposure_column}
        missing = required.difference(X.columns)
        if missing:
            raise ValueError(f"Colunas ausentes no baseline de contagem: {sorted(missing)}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (
            X[self.rate_column].to_numpy()
            * X[self.exposure_column].to_numpy()
        )


class ExposurePoissonRegressor(BaseEstimator, RegressorMixin):
    """Poisson para taxa com exposição fixa de população e horizonte."""

    def __init__(
        self,
        alpha: float = 0.1,
        feature_columns: list[str] | None = None,
        exposure_column: str = EXPOSURE_COLUMN,
    ):
        self.alpha = alpha
        self.feature_columns = feature_columns
        self.exposure_column = exposure_column

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ExposurePoissonRegressor":
        if self.exposure_column not in X:
            raise ValueError(f"A coluna de exposição {self.exposure_column} é obrigatória.")
        exposure = X[self.exposure_column].to_numpy(dtype=float)
        if np.any(exposure <= 0):
            raise ValueError("A exposição populacional deve ser positiva.")
        self.feature_names_ = list(self.feature_columns or [])
        if not self.feature_names_:
            self.feature_names_ = [
                column for column in X.columns if column != self.exposure_column
            ]
        missing = set(self.feature_names_).difference(X.columns)
        if missing:
            raise ValueError(f"Features ausentes no Poisson: {sorted(missing)}")
        response_rate = np.asarray(y, dtype=float) / exposure
        self.model_ = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    PoissonRegressor(alpha=self.alpha, max_iter=1000),
                ),
            ]
        )
        self.model_.fit(
            X[self.feature_names_],
            response_rate,
            model__sample_weight=exposure,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise ValueError("O modelo Poisson precisa ser ajustado antes da previsão.")
        exposure = X[self.exposure_column].to_numpy(dtype=float)
        predicted_rate = np.maximum(0, self.model_.predict(X[self.feature_names_]))
        return predicted_rate * exposure


def _prepare_panel(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    target_column, _, _ = smoothed_target_columns(horizon)
    required = {target_column, "populacao", "taxa_media_3_anos"}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no painel de contagem: {sorted(missing)}")
    result = panel.copy()
    result[EXPOSURE_COLUMN] = result["populacao"] * horizon / 100_000
    result[COUNT_TARGET] = result[target_column] * result[EXPOSURE_COLUMN]
    return result


def _count_metrics(
    split: pd.DataFrame,
    predicted_counts: np.ndarray,
    horizon: int,
) -> dict[str, float]:
    target_column, _, _ = smoothed_target_columns(horizon)
    predicted_counts = np.maximum(0, predicted_counts)
    exposure = split[EXPOSURE_COLUMN].to_numpy(dtype=float)
    predicted_rates = predicted_counts / exposure
    actual_rates = split[target_column].to_numpy(dtype=float)
    current_rates = split["taxa_media_3_anos"].to_numpy(dtype=float)
    actual_percent = 100 * (actual_rates - current_rates) / current_rates
    predicted_percent = 100 * (predicted_rates - current_rates) / current_rates
    return {
        "mae_contagem_proxy": float(
            mean_absolute_error(split[COUNT_TARGET], predicted_counts)
        ),
        "medae_contagem_proxy": float(
            median_absolute_error(split[COUNT_TARGET], predicted_counts)
        ),
        "mae_taxa": float(mean_absolute_error(actual_rates, predicted_rates)),
        "medae_taxa": float(median_absolute_error(actual_rates, predicted_rates)),
        "mae_percentual": float(
            mean_absolute_error(actual_percent, predicted_percent)
        ),
        "medae_percentual": float(
            median_absolute_error(actual_percent, predicted_percent)
        ),
    }


def _prediction_frame(
    split: pd.DataFrame,
    predicted_counts: np.ndarray,
    model_name: str,
    division: str,
    horizon: int,
) -> pd.DataFrame:
    target_column, _, _ = smoothed_target_columns(horizon)
    predicted_counts = np.maximum(0, predicted_counts)
    exposure = split[EXPOSURE_COLUMN].to_numpy(dtype=float)
    predicted_rates = predicted_counts / exposure
    current_rates = split["taxa_media_3_anos"].to_numpy(dtype=float)
    actual_rates = split[target_column].to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "municipio_codigo": split["municipio_codigo"].to_numpy(),
            "municipio": split["municipio"].to_numpy(),
            "ano": split["ano"].to_numpy(),
            "ano_futuro_inicio": split["ano_futuro_inicio"].to_numpy(),
            "ano_futuro": split["ano_futuro"].to_numpy(),
            "modelo": model_name,
            "divisao": division,
            "populacao": split["populacao"].to_numpy(),
            "exposicao_populacional_2_anos": exposure,
            "alvo_contagem_proxy_2_anos": split[COUNT_TARGET].to_numpy(),
            "contagem_proxy_2_anos_prevista": predicted_counts,
            "alvo_taxa_media_2_anos": actual_rates,
            "taxa_media_3_anos_atual": current_rates,
            "taxa_media_2_anos_prevista": predicted_rates,
            "alvo_variacao_media_2_anos_percentual": 100
            * (actual_rates - current_rates)
            / current_rates,
            "previsao_variacao_media_2_anos_percentual": 100
            * (predicted_rates - current_rates)
            / current_rates,
        }
    )


def run_count_sensitivity(
    panel: pd.DataFrame,
    output_dir: Path,
    model_root: Path,
    windows: list[tuple[int, int, int]],
    horizon: int = 2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Ajusta alpha na validação e avalia Poisson no teste temporal."""

    panel = _prepare_panel(panel, horizon)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output_dir / "painel_contagem_proxy_v11.csv", index=False)
    target_column, log_target_column, _ = smoothed_target_columns(horizon)
    del log_target_column
    tuning_rows: list[dict[str, Any]] = []
    cohorts = [
        ("principal_50mil", filter_by_population(panel)),
        ("todos", panel),
    ]
    for cohort, data in cohorts:
        for train_end, validation_end, test_end in windows:
            window = f"{validation_end + 1}-{test_end}"
            current = data[data.ano.le(test_end)].copy()
            splits = split_by_year(
                current, train_end=train_end, validation_end=validation_end
            )
            for alpha in ALPHA_GRID:
                model = ExposurePoissonRegressor(
                    alpha=alpha,
                    feature_columns=COUNT_FEATURES,
                )
                model.fit(splits["train"][COUNT_FEATURES + [EXPOSURE_COLUMN]], splits["train"][COUNT_TARGET])
                predicted = model.predict(
                    splits["validation"][COUNT_FEATURES + [EXPOSURE_COLUMN]]
                )
                metrics = _count_metrics(splits["validation"], predicted, horizon)
                tuning_rows.append(
                    {
                        "coorte": cohort,
                        "janela": window,
                        "alpha": alpha,
                        "n": len(splits["validation"]),
                        **metrics,
                    }
                )

    tuning = pd.DataFrame(tuning_rows)
    tuning["peso_mae_taxa"] = tuning["n"] * tuning["mae_taxa"]
    tuning_summary = (
        tuning.groupby("alpha", as_index=False)
        .agg(
            n_validacao=("n", "sum"),
            soma_peso_mae_taxa=("peso_mae_taxa", "sum"),
            mae_contagem_medio=("mae_contagem_proxy", "mean"),
            mae_percentual_medio=("mae_percentual", "mean"),
        )
    )
    tuning_summary["mae_taxa_ponderado"] = (
        tuning_summary["soma_peso_mae_taxa"] / tuning_summary["n_validacao"]
    )
    selected_alpha = float(
        tuning_summary.sort_values(
            ["mae_taxa_ponderado", "mae_contagem_medio", "alpha"]
        ).iloc[0]["alpha"]
    )
    tuning.drop(columns="peso_mae_taxa").to_csv(
        output_dir / "tuning_contagem_validacao.csv", index=False
    )
    tuning_summary.to_csv(output_dir / "tuning_contagem_resumo.csv", index=False)
    (output_dir / "parametro_poisson.json").write_text(
        json.dumps({"alpha": selected_alpha}, indent=2), encoding="utf-8"
    )

    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    choices: list[dict[str, Any]] = []
    for cohort, data in cohorts:
        for train_end, validation_end, test_end in windows:
            window = f"{validation_end + 1}-{test_end}"
            current = data[data.ano.le(test_end)].copy()
            splits = split_by_year(
                current, train_end=train_end, validation_end=validation_end
            )
            models = {
                "baseline_media_exposicao": MeanExposureCountRegressor(),
                "poisson_exposicao": ExposurePoissonRegressor(
                    alpha=selected_alpha,
                    feature_columns=COUNT_FEATURES,
                ),
            }
            model_dir = model_root / "contagem_poisson" / cohort / window
            model_dir.mkdir(parents=True, exist_ok=True)
            validation_metrics = []
            for model_name, model in models.items():
                model.fit(
                    splits["train"][COUNT_FEATURES + [EXPOSURE_COLUMN]],
                    splits["train"][COUNT_TARGET],
                )
                joblib.dump(model, model_dir / f"{model_name}.joblib")
                for division, split in splits.items():
                    predicted = model.predict(
                        split[COUNT_FEATURES + [EXPOSURE_COLUMN]]
                    )
                    metrics = {
                        "coorte": cohort,
                        "janela": window,
                        "modelo": model_name,
                        "divisao": division,
                        "n": len(split),
                        **_count_metrics(split, predicted, horizon),
                    }
                    metric_rows.append(metrics)
                    if division == "validation":
                        validation_metrics.append(metrics)
                    prediction_rows.append(
                        _prediction_frame(split, predicted, model_name, division, horizon).assign(
                            coorte=cohort, janela=window
                        )
                    )
            choice = (
                pd.DataFrame(validation_metrics)
                .sort_values(["mae_taxa", "mae_contagem_proxy", "modelo"])
                .iloc[0]
                .to_dict()
            )
            choices.append(choice)

    metrics_frame = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    choices_frame = pd.DataFrame(choices)
    selected = predictions.merge(
        choices_frame[["coorte", "janela", "modelo"]],
        on=["coorte", "janela", "modelo"],
        how="inner",
        validate="many_to_one",
    )
    metrics_frame.to_csv(output_dir / "metricas_contagem_v11.csv", index=False)
    predictions[predictions.divisao.eq("test")].to_csv(
        output_dir / "previsoes_contagem_v11_teste.csv", index=False
    )
    choices_frame.to_csv(
        output_dir / "selecao_contagem_v11_validacao.csv", index=False
    )
    selected[selected.divisao.eq("test")].to_csv(
        output_dir / "previsoes_contagem_v11_mapa.csv", index=False
    )
    return {
        "alpha": selected_alpha,
        "metrics": metrics_frame,
        "predictions": predictions,
        "choices": choices_frame,
    }
