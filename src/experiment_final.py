"""Execução retrospectiva do MVP final."""

import json

from .count_sensibilidade import run_count_sensitivity
from .enrich_previsoes import enrich_predictions
from .experiment_core import RAW, ROOT, run_experiment
from .indicators import load_atlas_rates, municipality_lookup
from .population import load_population
from .prepare_data import FEATURE_COLUMNS_FINAL, build_smoothed_model_dataset
from .tune_temporal import tune_rate_models


OUT = ROOT / "data/processed/v11"
MODEL_ROOT = ROOT / "models/v11"
WINDOWS_V11 = [
    (2012, 2014, 2016),
    (2014, 2016, 2018),
    (2016, 2018, 2020),
]


def _load_panel():
    population = load_population(RAW / "populacao_municipal_ibge_2001_2021.csv")
    lookup = municipality_lookup(population, RAW / "municipios_ibge.json")
    rates = load_atlas_rates(
        RAW / "Taxa de homicídios registrados_municípios.csv", lookup
    )
    return build_smoothed_model_dataset(
        rates,
        population,
        horizon=2,
        history_years=5,
    )


def main() -> None:
    panel = _load_panel()
    selected_params = tune_rate_models(
        panel,
        output_dir=OUT,
        windows=WINDOWS_V11,
        horizon=2,
        feature_columns=FEATURE_COLUMNS_FINAL,
    )
    run_experiment(
        output_dir=OUT,
        model_root=MODEL_ROOT,
        version=11,
        horizon=2,
        windows=WINDOWS_V11,
        experiment="atlas_populacao_historico_5_anos_tuning_temporal",
        history_years=5,
        feature_columns=FEATURE_COLUMNS_FINAL,
        model_params=selected_params,
    )
    count_result = run_count_sensitivity(
        panel,
        output_dir=OUT,
        model_root=MODEL_ROOT,
        windows=WINDOWS_V11,
        horizon=2,
    )
    enrich_predictions(
        output_dir=OUT,
        model_root=MODEL_ROOT,
        panel=panel,
        windows=WINDOWS_V11,
        horizon=2,
        confidence=0.90,
    )
    config_path = OUT / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["count_model"] = {
        "habilitado": True,
        "modelo": "Poisson com exposição populacional",
        "alpha_selecionado_validacao": count_result["alpha"],
        "alvo": "contagem proxy = taxa média futura × população × horizonte / 100000",
        "observacao": "A contagem é derivada da taxa do Atlas e não é numerador oficial.",
    }
    config["representacao_mapa"] = {
        "taxa_futura_media": True,
        "probabilidade_quartil_superior_calibrada": True,
        "faixas_risco": ["baixo", "moderado", "alto", "muito_alto"],
        "limites_faixas_probabilidade": [0.25, 0.50, 0.75],
        "ranking_relativo": "ordem decrescente da taxa futura prevista por coorte, janela e ano",
        "intervalo": "conformal de 90%, quantil absoluto ajustado somente na validação",
    }
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
