"""Executa o experimento do MVP analítico e salva seus artefatos."""

import json
import platform
from pathlib import Path

import numpy
import pandas
import sklearn

from .models import evaluate_by_population_size, run_experiment
from .population import filter_by_population, load_population
from .prepare_data import (
    TARGET_LOG_COLUMN,
    build_model_dataset,
    get_feature_columns,
    load_homicide_rates,
)


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "taxa_homicidios_2024.csv"
POPULATION_PATH = ROOT / "data" / "raw" / "populacao_municipal_ibge_2001_2021.csv"
PROCESSED_PATH = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "models"
MIN_POPULATION = 50_000


def main() -> None:
    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    rates = load_homicide_rates(RAW_PATH)
    population = load_population(POPULATION_PATH)
    full_model_data = build_model_dataset(rates, population=population)
    model_data = filter_by_population(full_model_data, minimum=MIN_POPULATION)
    full_metrics, full_predictions = run_experiment(full_model_data)
    metrics, predictions = run_experiment(model_data, model_output_dir=MODEL_PATH)
    size_metrics = evaluate_by_population_size(predictions)

    full_model_data.to_csv(PROCESSED_PATH / "model_dataset_todos.csv", index=False)
    model_data.to_csv(PROCESSED_PATH / "model_dataset.csv", index=False)
    full_metrics.to_csv(PROCESSED_PATH / "model_metrics_todos.csv", index=False)
    metrics.to_csv(PROCESSED_PATH / "model_metrics.csv", index=False)
    full_predictions.to_csv(
        PROCESSED_PATH / "model_predictions_todos.csv", index=False
    )
    predictions.to_csv(PROCESSED_PATH / "model_predictions.csv", index=False)
    size_metrics.to_csv(PROCESSED_PATH / "model_metrics_by_porte.csv", index=False)

    config = {
        "random_state": 42,
        "train_end": 2016,
        "validation_end": 2019,
        "minimum_population": MIN_POPULATION,
        "population_filter": "populacao no ano de origem",
        "feature_columns": get_feature_columns(model_data),
        "target_column": TARGET_LOG_COLUMN,
        "population_source": "IBGE SIDRA tabela 6579; tabela 579 para 2007; tabela 202 para 2010",
        "versions": {
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    (MODEL_PATH / "model_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    test_metrics = metrics[metrics["divisao"].eq("test")].sort_values(
        "mae_log_100"
    )
    print(f"Observações preparadas: {len(model_data):,}")
    print(f"Observações na base completa: {len(full_model_data):,}")
    print(
        "População observada: "
        f"{model_data['populacao_observada'].mean():.1%}"
    )
    print("Métricas no teste:")
    print(test_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
