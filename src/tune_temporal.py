"""Ajuste temporal de hiperparâmetros do modelo final."""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .experiment_core import ROOT
from .models_final import (
    _authorized_features,
    build_smoothed_rate_models,
    evaluate_smoothed_rate_predictions,
)
from .population import filter_by_population
from .prepare_data import FEATURE_COLUMNS_FINAL, split_by_year, smoothed_target_columns


TUNING_GRIDS: dict[str, list[dict[str, Any]]] = {
    "ridge": [
        {"alpha": 0.1},
        {"alpha": 1.0},
        {"alpha": 10.0},
    ],
    "random_forest": [
        {"n_estimators": 200, "min_samples_leaf": 5, "max_features": "sqrt"},
        {"n_estimators": 200, "min_samples_leaf": 10, "max_features": "sqrt"},
        {"n_estimators": 200, "min_samples_leaf": 20, "max_features": "sqrt"},
        {"n_estimators": 200, "min_samples_leaf": 5, "max_features": 1.0},
        {"n_estimators": 200, "min_samples_leaf": 10, "max_features": 1.0},
        {"n_estimators": 200, "min_samples_leaf": 20, "max_features": 1.0},
    ],
    "hist_gradient_boosting": [
        {"max_iter": 100, "learning_rate": 0.05, "max_leaf_nodes": 15},
        {"max_iter": 200, "learning_rate": 0.05, "max_leaf_nodes": 15},
        {"max_iter": 100, "learning_rate": 0.1, "max_leaf_nodes": 15},
        {"max_iter": 200, "learning_rate": 0.1, "max_leaf_nodes": 15},
    ],
}


def tune_rate_models(
    panel: pd.DataFrame,
    output_dir: Path,
    windows: list[tuple[int, int, int]],
    horizon: int = 2,
    feature_columns: list[str] = FEATURE_COLUMNS_FINAL,
    random_state: int = 42,
) -> dict[str, dict[str, Any]]:
    """Escolhe uma configuração global por família usando apenas validação.

    Cada candidato é ajustado no treino da janela e pontuado na validação. O
    conjunto de teste não é previsto nem consultado nesta função.
    """

    target_column, log_target_column, _ = smoothed_target_columns(horizon)
    rows: list[dict[str, Any]] = []
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
            features = _authorized_features(current, feature_columns)
            validation = splits["validation"]
            for model_name, candidates in TUNING_GRIDS.items():
                for candidate_index, params in enumerate(candidates, start=1):
                    model = build_smoothed_rate_models(
                        random_state=random_state,
                        horizon=horizon,
                        model_params={model_name: params},
                    )[model_name]
                    model.fit(
                        splits["train"][features],
                        splits["train"][log_target_column],
                    )
                    predicted = model.predict(validation[features])
                    metrics = evaluate_smoothed_rate_predictions(
                        validation[log_target_column],
                        predicted,
                        validation[target_column],
                        validation["taxa_media_3_anos"],
                    )
                    rows.append(
                        {
                            "coorte": cohort,
                            "janela": window,
                            "modelo": model_name,
                            "candidato": candidate_index,
                            "parametros": json.dumps(params, sort_keys=True),
                            "n": len(validation),
                            **metrics,
                        }
                    )

    detail = pd.DataFrame(rows)
    if detail.empty:
        raise ValueError("Nenhum candidato foi avaliado na validação.")
    detail["peso_mae_taxa"] = detail["n"] * detail["mae_taxa"]
    summary = (
        detail.groupby(["modelo", "candidato", "parametros"], as_index=False)
        .agg(
            n_validacao=("n", "sum"),
            soma_peso_mae_taxa=("peso_mae_taxa", "sum"),
            mae_log_100_medio=("mae_log_100", "mean"),
            mae_percentual_medio=("mae_percentual", "mean"),
        )
    )
    summary["mae_taxa_ponderado"] = (
        summary["soma_peso_mae_taxa"] / summary["n_validacao"]
    )
    selected = (
        summary.sort_values(
            ["modelo", "mae_taxa_ponderado", "mae_log_100_medio", "candidato"]
        )
        .groupby("modelo", as_index=False)
        .head(1)
        .sort_values("modelo")
    )
    selected_params = {
        row.modelo: json.loads(row.parametros)
        for row in selected.itertuples()
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    detail.drop(columns="peso_mae_taxa").to_csv(
        output_dir / "tuning_taxa_validacao.csv", index=False
    )
    summary.to_csv(output_dir / "tuning_taxa_resumo.csv", index=False)
    selected.to_csv(output_dir / "parametros_taxa_selecionados.csv", index=False)
    (output_dir / "parametros_modelos.json").write_text(
        json.dumps(selected_params, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return selected_params
