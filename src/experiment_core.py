"""Rotinas compartilhadas pelos experimentos do MVP final."""

import hashlib
import json
from datetime import date
from pathlib import Path

import pandas as pd

from .indicators import load_atlas_rates, municipality_lookup
from .models_final import (
    run_risk_classification_experiment,
    run_smoothed_rate_experiment,
)
from .population import filter_by_population, load_population
from .prepare_data import build_smoothed_model_dataset

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"

SOURCE_METADATA = [
    {
        "arquivo": "data/raw/Taxa de homicídios registrados_municípios.csv",
        "produtor": "Ipea/FBSP — Atlas da Violência",
        "url": "https://www.ipea.gov.br/atlasviolencia/tema/1/",
        "obtencao": None,
        "proveniencia": "exportacao local fornecida pelo autor; URL e data exatas da exportacao nao registradas",
    },
    {
        "arquivo": "data/raw/populacao_municipal_ibge_2001_2021.csv",
        "produtor": "IBGE/SIDRA — consolidacao local",
        "url": "https://sidra.ibge.gov.br/tabela/6579; https://sidra.ibge.gov.br/tabela/579; https://sidra.ibge.gov.br/tabela/202",
        "obtencao": "2026-08-19",
        "proveniencia": "arquivo municipal consolidado a partir das tabelas listadas",
    },
    {
        "arquivo": "data/raw/municipios_ibge.json",
        "produtor": "IBGE",
        "url": "https://servicodados.ibge.gov.br/api/v1/localidades/municipios",
        "obtencao": "2026-09-27",
        "proveniencia": "cadastro de nomes e codigos preservado localmente",
    },
]


def _run_rate_panel(
    panel: pd.DataFrame,
    model_root: Path,
    horizon: int,
    windows: list[tuple[int, int, int]],
    experiment: str,
    feature_columns: list[str],
    include_ensemble: bool = False,
    model_params: dict[str, dict] | None = None,
):
    metrics_all, predictions_all, choices_all = [], [], []
    for cohort, data in [
        ("principal_50mil", filter_by_population(panel)),
        ("todos", panel),
    ]:
        for train_end, validation_end, test_end in windows:
            window = f"{validation_end + 1}-{test_end}"
            current = data[data.ano.le(test_end)].copy()
            model_dir = model_root / "taxa_suavizada" / cohort / window
            metrics, predictions = run_smoothed_rate_experiment(
                current,
                train_end=train_end,
                validation_end=validation_end,
                feature_columns=feature_columns,
                horizon=horizon,
                include_ensemble=include_ensemble,
                model_params=model_params,
                model_output_dir=model_dir,
            )
            tags = dict(coorte=cohort, experimento=experiment, janela=window)
            metrics = metrics.assign(**tags)
            predictions = predictions.assign(**tags)
            choice = (
                metrics[metrics.divisao.eq("validation")]
                .sort_values(["mae_taxa", "mae_log_100", "modelo"])
                .head(1)
            )
            metrics_all.append(metrics)
            predictions_all.append(predictions[predictions.divisao.eq("test")])
            choices_all.append(choice)

    return (
        pd.concat(metrics_all, ignore_index=True),
        pd.concat(predictions_all, ignore_index=True),
        pd.concat(choices_all, ignore_index=True),
    )


def _run_classification_panel(
    panel: pd.DataFrame,
    model_root: Path,
    horizon: int,
    windows: list[tuple[int, int, int]],
    experiment: str,
    feature_columns: list[str],
):
    metrics_all, predictions_all, choices_all = [], [], []
    for cohort, data in [
        ("principal_50mil", filter_by_population(panel)),
        ("todos", panel),
    ]:
        for train_end, validation_end, test_end in windows:
            window = f"{validation_end + 1}-{test_end}"
            current = data[data.ano.le(test_end)].copy()
            model_dir = model_root / "risco_alto" / cohort / window
            metrics, predictions, _ = run_risk_classification_experiment(
                current,
                train_end=train_end,
                validation_end=validation_end,
                feature_columns=feature_columns,
                horizon=horizon,
                model_output_dir=model_dir,
            )
            tags = dict(coorte=cohort, experimento=experiment, janela=window)
            metrics = metrics.assign(**tags)
            predictions = predictions.assign(**tags)
            choice = (
                metrics[metrics.divisao.eq("validation")]
                .sort_values(
                    ["balanced_accuracy", "f1", "roc_auc", "modelo"],
                    ascending=[False, False, False, True],
                )
                .head(1)
            )
            metrics_all.append(metrics)
            predictions_all.append(predictions[predictions.divisao.eq("test")])
            choices_all.append(choice)

    return (
        pd.concat(metrics_all, ignore_index=True),
        pd.concat(predictions_all, ignore_index=True),
        pd.concat(choices_all, ignore_index=True),
    )


def _selected_predictions(
    predictions: pd.DataFrame, choices: pd.DataFrame
) -> pd.DataFrame:
    keys = ["coorte", "experimento", "janela", "modelo"]
    return predictions.merge(
        choices[keys], on=keys, how="inner", validate="many_to_one"
    )


def _save_config(
    panel: pd.DataFrame,
    output_dir: Path,
    version: int,
    horizon: int,
    windows: list[tuple[int, int, int]],
    feature_columns: list[str],
    include_ensemble: bool = False,
    model_params: dict[str, dict] | None = None,
) -> None:
    sources = []
    for source in SOURCE_METADATA:
        path = ROOT / source["arquivo"]
        if not path.exists():
            continue
        item = dict(source)
        item["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        item["bytes"] = path.stat().st_size
        sources.append(item)

    config = {
        "versao": version,
        "horizonte_anos": horizon,
        "gerado_em": date.today().isoformat(),
        "fonte_principal": "Atlas da Violência - taxa municipal",
        "fontes_complementares": [
            "IBGE - população municipal",
            "IBGE - cadastro de municípios",
        ],
        "alvo_regressao": f"taxa_futura_media_{horizon}_anos",
        "definicao_alvo_regressao": f"media das taxas Atlas em t+1 ate t+{horizon}",
        "alvo_classificacao": f"quartil superior da taxa futura média de {horizon} anos",
        "limiar_classificacao": "quantil 0.75 calculado somente no treino de cada janela",
        "probabilidade_classificacao_calibrada": True,
        "metodo_calibracao_classificacao": "isotônica ajustada somente na validação de cada janela",
        "features": feature_columns,
        "coorte_principal": "populacao no ano de origem >= 50.000",
        "janelas": windows,
        "selecao_regressao": "menor MAE de taxa na validacao",
        "selecao_classificacao": "maior balanced accuracy na validacao",
        "ensemble": {
            "habilitado": include_ensemble,
            "metodo": "media aritmetica na escala logaritmica",
            "membros": [
                f"baseline_media_{horizon}_anos",
                "random_forest",
                "hist_gradient_boosting",
            ],
        },
        "model_params": model_params or {},
        "linhas_painel": len(panel),
        "fontes": sources,
        "fontes_sha256": {
            item["arquivo"]: item["sha256"] for item in sources
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run_experiment(
    output_dir: Path,
    model_root: Path,
    version: int,
    horizon: int,
    windows: list[tuple[int, int, int]],
    experiment: str,
    history_years: int,
    feature_columns: list[str],
    include_ensemble: bool = False,
    model_params: dict[str, dict] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    population = load_population(RAW / "populacao_municipal_ibge_2001_2021.csv")
    lookup = municipality_lookup(population, RAW / "municipios_ibge.json")
    rates = load_atlas_rates(
        RAW / "Taxa de homicídios registrados_municípios.csv", lookup
    )
    panel = build_smoothed_model_dataset(
        rates,
        population,
        horizon=horizon,
        history_years=history_years,
    )
    panel.to_csv(output_dir / "painel.csv", index=False)
    _save_config(
        panel,
        output_dir=output_dir,
        version=version,
        horizon=horizon,
        windows=windows,
        feature_columns=feature_columns,
        include_ensemble=include_ensemble,
        model_params=model_params,
    )

    rate_metrics, rate_predictions, rate_choices = _run_rate_panel(
        panel,
        model_root=model_root,
        horizon=horizon,
        windows=windows,
        experiment=experiment,
        feature_columns=feature_columns,
        include_ensemble=include_ensemble,
        model_params=model_params,
    )
    rate_metrics.to_csv(output_dir / "metricas_taxa_suavizada_janelas.csv", index=False)
    rate_predictions.to_csv(output_dir / "previsoes_taxa_suavizada_teste.csv", index=False)
    rate_choices.to_csv(output_dir / "selecao_taxa_suavizada_validacao.csv", index=False)
    selected_rate = _selected_predictions(rate_predictions, rate_choices)
    selected_rate.to_csv(output_dir / "previsoes_taxa_suavizada_mapa.csv", index=False)

    risk_metrics, risk_predictions, risk_choices = _run_classification_panel(
        panel,
        model_root=model_root,
        horizon=horizon,
        windows=windows,
        experiment=experiment,
        feature_columns=feature_columns,
    )
    risk_metrics.to_csv(output_dir / "metricas_risco_alto_janelas.csv", index=False)
    risk_predictions.to_csv(output_dir / "previsoes_risco_alto_teste.csv", index=False)
    risk_choices.to_csv(output_dir / "selecao_risco_alto_validacao.csv", index=False)
    selected_risk = _selected_predictions(risk_predictions, risk_choices)
    selected_risk.to_csv(output_dir / "previsoes_risco_alto_mapa.csv", index=False)

    risk_target_column = f"taxa_futura_media_{horizon}_anos"
    map_data = selected_rate.merge(
        selected_risk[
            [
                "coorte",
                "experimento",
                "janela",
                "municipio_codigo",
                "ano",
                "modelo",
                risk_target_column,
                "risco_alto_observado",
                "probabilidade_alto_risco",
                "probabilidade_alto_risco_calibrada",
                "risco_alto_previsto",
                "risco_alto_previsto_calibrado",
                "limiar_taxa_alto_risco",
            ]
        ],
        on=["coorte", "experimento", "janela", "municipio_codigo", "ano"],
        how="left",
        suffixes=("_regressao", "_classificacao"),
        validate="one_to_one",
    )
    map_data["uf_codigo"] = map_data["municipio_codigo"].str[:2]
    map_data.to_csv(output_dir / "previsoes_mapa.csv", index=False)
