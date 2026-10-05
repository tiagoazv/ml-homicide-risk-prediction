"""Geração da previsão futura 2024--2026 com os artefatos finais."""

import hashlib
import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .enrich_previsoes import RISK_BINS, RISK_LABELS, _conformal_quantile
from .indicators import load_atlas_rates, municipality_lookup
from .models_final import (
    _authorized_features,
    build_risk_classifiers,
    build_smoothed_rate_models,
    calibrate_validation_probabilities,
)
from .population import (
    attach_population_features,
    combine_population_sources,
    filter_by_population,
    load_population,
    load_sidra_population_json,
)
from .prepare_data import (
    FEATURE_COLUMNS_FINAL,
    build_smoothed_model_dataset,
    smoothed_target_columns,
)


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
OUT = ROOT / "data/processed/v11"
MODEL_ROOT = ROOT / "models/v11/futuro_2024_2026"
ORIGIN_YEAR = 2024
HORIZON = 2
HISTORY_YEARS = 5
CALIBRATION_WINDOW = (2016, 2018, 2020)
FUTURE_EXPERIMENT = "atlas_populacao_mvp_final_previsao_futura_2024_2026"


def load_extended_population(raw_dir: Path = RAW) -> pd.DataFrame:
    """Combina a série histórica com as populações municipais do SIDRA."""

    historical = load_population(raw_dir / "populacao_municipal_ibge_2001_2021.csv")
    census_2022 = load_sidra_population_json(raw_dir / "sidra_populacao_2022.json")
    estimate_2024 = load_sidra_population_json(raw_dir / "sidra_populacao_2024.json")
    return combine_population_sources(historical, census_2022, estimate_2024)


def _add_historical_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Calcula as mesmas features históricas usadas pelo modelo final."""

    data = frame.copy().sort_values(["municipio_codigo", "ano"])
    grouped = data.groupby("municipio_codigo", sort=False)
    for offset in range(1, HISTORY_YEARS):
        data[f"taxa_lag_{offset}"] = grouped["taxa_homicidios"].shift(offset)
        data[f"ano_lag_{offset}"] = grouped["ano"].shift(offset)

    interval_1 = data["ano"].sub(data["ano_lag_1"])
    interval_2 = data["ano_lag_1"].sub(data["ano_lag_2"])
    interval_3 = data["ano_lag_2"].sub(data["ano_lag_3"])
    interval_4 = data["ano_lag_3"].sub(data["ano_lag_4"])
    valid_3 = interval_1.eq(1).fillna(False) & interval_2.eq(1).fillna(False)
    valid_5 = valid_3 & interval_3.eq(1).fillna(False) & interval_4.eq(1).fillna(False)

    history_3 = data[["taxa_lag_2", "taxa_lag_1", "taxa_homicidios"]]
    history_5 = data[
        ["taxa_lag_4", "taxa_lag_3", "taxa_lag_2", "taxa_lag_1", "taxa_homicidios"]
    ]
    data["taxa_media_3_anos"] = np.where(valid_3, history_3.mean(axis=1), np.nan)
    data["taxa_std_3_anos"] = np.where(
        valid_3, history_3.std(axis=1, ddof=0), np.nan
    )
    data["taxa_mediana_3_anos"] = np.where(valid_3, history_3.median(axis=1), np.nan)
    data["tendencia_log_3_anos"] = np.where(
        valid_3,
        (np.log1p(data["taxa_homicidios"]) - np.log1p(data["taxa_lag_2"])) / 2 * 100,
        np.nan,
    )
    data["variacao_log_100_atual"] = np.where(
        interval_1.eq(1),
        100 * (np.log1p(data["taxa_homicidios"]) - np.log1p(data["taxa_lag_1"])),
        np.nan,
    )
    data["aumentos_3_anos"] = np.where(
        valid_3,
        data["taxa_lag_1"].gt(data["taxa_lag_2"]).astype(int)
        + data["taxa_homicidios"].gt(data["taxa_lag_1"]).astype(int),
        np.nan,
    )
    data["taxa_media_5_anos"] = np.where(valid_5, history_5.mean(axis=1), np.nan)
    data["taxa_std_5_anos"] = np.where(
        valid_5, history_5.std(axis=1, ddof=0), np.nan
    )
    data["taxa_mediana_5_anos"] = np.where(valid_5, history_5.median(axis=1), np.nan)
    data["amplitude_taxa_5_anos"] = np.where(
        valid_5, history_5.max(axis=1) - history_5.min(axis=1), np.nan
    )
    data["tendencia_taxa_5_anos"] = np.where(
        valid_5, (data["taxa_homicidios"] - data["taxa_lag_4"]) / 4, np.nan
    )
    data["tendencia_log_5_anos"] = np.where(
        valid_5,
        (np.log1p(data["taxa_homicidios"]) - np.log1p(data["taxa_lag_4"])) / 4 * 100,
        np.nan,
    )
    data["aumentos_5_anos"] = np.where(
        valid_5,
        data["taxa_lag_3"].gt(data["taxa_lag_4"]).astype(int)
        + data["taxa_lag_2"].gt(data["taxa_lag_3"]).astype(int)
        + data["taxa_lag_1"].gt(data["taxa_lag_2"]).astype(int)
        + data["taxa_homicidios"].gt(data["taxa_lag_1"]).astype(int),
        np.nan,
    )
    data["_historico_valido"] = valid_5
    return data


def build_future_feature_panel(
    rates: pd.DataFrame,
    population: pd.DataFrame,
    origin_year: int = ORIGIN_YEAR,
) -> pd.DataFrame:
    """Cria as features da origem futura sem fabricar qualquer alvo."""

    data = _add_historical_features(rates)
    data = data.loc[data.ano.eq(origin_year) & data._historico_valido].copy()
    if data.empty:
        raise ValueError(f"Não há histórico consecutivo de cinco anos para {origin_year}.")
    data = attach_population_features(data, population)
    data["ano_futuro_inicio"] = origin_year + 1
    data["ano_futuro_final"] = origin_year + HORIZON
    data["ano_futuro"] = origin_year + HORIZON
    data = data.drop(columns="_historico_valido", errors="ignore")
    return data.reset_index(drop=True)


def _latest_selection(path: Path, cohort: str) -> pd.Series:
    selected = pd.read_csv(path)
    selected = selected.loc[selected.coorte.eq(cohort)].copy()
    if selected.empty:
        raise ValueError(f"Não há seleção histórica para a coorte {cohort}.")
    selected["fim_janela"] = selected.janela.str.split("-").str[-1].astype(int)
    return selected.sort_values("fim_janela").iloc[-1]


def _new_rate_model(model_name: str, params: dict[str, dict]):
    models = build_smoothed_rate_models(
        horizon=HORIZON,
        model_params=params,
    )
    if model_name not in models:
        raise ValueError(f"Modelo de taxa não disponível: {model_name}")
    return models[model_name]


def _rate_calibration_quantile(
    data: pd.DataFrame,
    model_name: str,
    params: dict[str, dict],
    features: list[str],
) -> float:
    train_end, validation_end, _ = CALIBRATION_WINDOW
    train = data.loc[data.ano.le(train_end)]
    validation = data.loc[data.ano.gt(train_end) & data.ano.le(validation_end)]
    _, target_log, _ = smoothed_target_columns(HORIZON)
    if train.empty or validation.empty:
        raise ValueError("A janela de calibração futura ficou vazia.")
    model = _new_rate_model(model_name, params)
    model.fit(train[features], train[target_log])
    residuals = np.abs(validation[target_log].to_numpy() - model.predict(validation[features]))
    return _conformal_quantile(residuals, 0.90)


def _risk_calibration(
    data: pd.DataFrame,
    model_name: str,
    features: list[str],
) -> tuple[object, float, np.ndarray, pd.Series, bool]:
    train_end, validation_end, _ = CALIBRATION_WINDOW
    train = data.loc[data.ano.le(train_end)]
    validation = data.loc[data.ano.gt(train_end) & data.ano.le(validation_end)]
    target, _, _ = smoothed_target_columns(HORIZON)
    threshold = float(train[target].quantile(0.75))
    train_observed = train[target].ge(threshold).astype(int)
    if train_observed.nunique() < 2 or validation.empty:
        raise ValueError("A calibração de risco não possui duas classes ou validação.")
    calibration_model = build_risk_classifiers()[model_name]
    calibration_model.fit(train[features], train_observed)
    validation_probability = calibration_model.predict_proba(validation[features])[:, 1]
    validation_observed = validation[target].ge(threshold).astype(int)
    return (
        calibration_model,
        threshold,
        validation_probability,
        validation_observed,
        validation_observed.nunique() >= 2,
    )


def _rank_and_risk(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    group = ["coorte", "janela", "ano"]
    result["ranking_taxa_futura"] = result.groupby(group)["taxa_prevista"].rank(
        method="min", ascending=False
    ).astype("Int64")
    counts = result.groupby(group)["taxa_prevista"].transform("count")
    result["percentil_ranking_taxa"] = np.where(
        counts.gt(1), 1 - (result["ranking_taxa_futura"] - 1) / (counts - 1), 1.0
    )
    result["faixa_risco"] = pd.cut(
        result["probabilidade_alto_risco_calibrada"],
        bins=RISK_BINS,
        labels=RISK_LABELS,
        include_lowest=True,
    ).astype("string")
    return result


def _source_metadata(raw_dir: Path) -> list[dict[str, object]]:
    files = [
        ("data/raw/Taxa de homicídios registrados_municípios.csv", "Atlas da Violência"),
        ("data/raw/populacao_municipal_ibge_2001_2021.csv", "IBGE/SIDRA"),
        ("data/raw/sidra_populacao_2022.json", "IBGE/SIDRA"),
        ("data/raw/sidra_populacao_2024.json", "IBGE/SIDRA"),
        ("data/raw/municipios_ibge.json", "IBGE"),
    ]
    result = []
    for relative, producer in files:
        path = raw_dir / relative.removeprefix("data/raw/")
        result.append(
            {
                "arquivo": relative,
                "produtor": producer,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
        )
    return result


def run_future_forecast(
    output_dir: Path = OUT,
    model_root: Path = MODEL_ROOT,
    raw_dir: Path = RAW,
    origin_year: int = ORIGIN_YEAR,
) -> pd.DataFrame:
    """Treina com alvos históricos e grava a previsão futura não avaliada."""

    population = load_extended_population(raw_dir)
    lookup = municipality_lookup(population, raw_dir / "municipios_ibge.json")
    rates = load_atlas_rates(
        raw_dir / "Taxa de homicídios registrados_municípios.csv", lookup
    )
    panel = build_smoothed_model_dataset(
        rates, population, horizon=HORIZON, history_years=HISTORY_YEARS
    )
    future = build_future_feature_panel(rates, population, origin_year=origin_year)
    params = json.loads((output_dir / "parametros_modelos.json").read_text())
    rate_selection_path = output_dir / "selecao_taxa_suavizada_validacao.csv"
    risk_selection_path = output_dir / "selecao_risco_alto_validacao.csv"
    target, target_log, _ = smoothed_target_columns(HORIZON)
    features = _authorized_features(panel, FEATURE_COLUMNS_FINAL)
    output_parts = []
    selected_models = {}

    for cohort, historical in [
        ("principal_50mil", filter_by_population(panel)),
        ("todos", panel),
    ]:
        current = filter_by_population(future) if cohort == "principal_50mil" else future
        rate_choice = _latest_selection(rate_selection_path, cohort)
        risk_choice = _latest_selection(risk_selection_path, cohort)
        rate_model_name = str(rate_choice.modelo)
        risk_model_name = str(risk_choice.modelo)

        rate_model = _new_rate_model(rate_model_name, params)
        rate_model.fit(historical[features], historical[target_log])
        model_dir = model_root / cohort
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(rate_model, model_dir / "taxa.joblib")
        predicted_log = rate_model.predict(current[features])
        predicted_rate = np.maximum(0, np.expm1(predicted_log / 100))

        quantile = _rate_calibration_quantile(
            historical, rate_model_name, params, features
        )
        lower = np.maximum(0, np.expm1((predicted_log - quantile) / 100))
        upper = np.maximum(0, np.expm1((predicted_log + quantile) / 100))

        _, threshold, validation_probability, validation_observed, can_calibrate = (
            _risk_calibration(historical, risk_model_name, features)
        )
        risk_model = build_risk_classifiers()[risk_model_name]
        risk_target = historical[target].ge(threshold).astype(int)
        risk_model.fit(historical[features], risk_target)
        joblib.dump(risk_model, model_dir / "risco.joblib")
        probability = risk_model.predict_proba(current[features])[:, 1]
        calibrated_probability, calibration_applied = calibrate_validation_probabilities(
            validation_probability,
            validation_observed,
            probability,
        ) if can_calibrate else (probability, False)

        part = current[
            [
                "municipio_codigo",
                "municipio",
                "uf_codigo",
                "ano",
                "populacao",
                "porte_populacional",
                "taxa_media_3_anos",
            ]
        ].copy()
        part["ano_futuro_inicio"] = origin_year + 1
        part["ano_futuro_final"] = origin_year + HORIZON
        part["ano_futuro"] = origin_year + HORIZON
        part["modelo"] = rate_model_name
        part["modelo_regressao"] = rate_model_name
        part["modelo_classificacao"] = risk_model_name
        part["divisao"] = "futuro"
        part["experimento"] = FUTURE_EXPERIMENT
        part["janela"] = f"{origin_year}-{origin_year + HORIZON}"
        part["coorte"] = cohort
        part["taxa_atual"] = part["taxa_media_3_anos"]
        part["taxa_media_3_anos_atual"] = part["taxa_media_3_anos"]
        part["taxa_prevista"] = predicted_rate
        part[f"taxa_media_{HORIZON}_anos_prevista"] = predicted_rate
        part["taxa_media_futura_prevista"] = predicted_rate
        part["previsao_log_taxa_media_2_anos_100"] = predicted_log
        part["previsao_percentual"] = np.where(
            part["taxa_atual"].gt(0),
            (predicted_rate - part["taxa_atual"]) / part["taxa_atual"] * 100,
            np.nan,
        )
        part["previsao_variacao_media_futura_percentual"] = part["previsao_percentual"]
        part["probabilidade_alto_risco"] = probability
        part["probabilidade_alto_risco_calibrada"] = calibrated_probability
        part["risco_alto_previsto_calibrado"] = (calibrated_probability >= 0.5).astype(int)
        part["limiar_taxa_alto_risco"] = threshold
        part["quantil_conformal_log_100"] = quantile
        part[f"intervalo_inferior_taxa_media_{HORIZON}_anos"] = lower
        part[f"intervalo_superior_taxa_media_{HORIZON}_anos"] = upper
        part["exposicao_populacional_2_anos"] = part["populacao"] * 2 / 100_000
        part["homicidios_proxy_futuro_2_anos"] = (
            predicted_rate * part["exposicao_populacional_2_anos"]
        )
        part["situacao"] = "Previsão futura não avaliada"
        part["previsao_tipo"] = "futura_nao_avaliada"
        output_parts.append(part)
        selected_models[cohort] = {
            "modelo_taxa": rate_model_name,
            "modelo_risco": risk_model_name,
            "janela_selecao_taxa": str(rate_choice.janela),
            "janela_selecao_risco": str(risk_choice.janela),
            "limiar_risco": threshold,
            "quantil_conformal_log_100": quantile,
            "calibracao_risco_aplicada": calibration_applied,
            "n_historico_treino": int(len(historical)),
            "n_previsoes": int(len(current)),
        }

    result = _rank_and_risk(pd.concat(output_parts, ignore_index=True))
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "previsoes_futuras_2024_2026.csv", index=False)
    config = {
        "versao": 11,
        "tipo": "previsao_futura_nao_avaliada",
        "gerado_em": date.today().isoformat(),
        "origem": origin_year,
        "horizonte_anos": HORIZON,
        "anos_previstos": [origin_year + 1, origin_year + 2],
        "historico_taxas_exigido": list(range(origin_year - HISTORY_YEARS + 1, origin_year + 1)),
        "maximo_ano_com_alvo_disponivel_no_treino": int(panel.ano.max()),
        "features": FEATURE_COLUMNS_FINAL,
        "model_params": params,
        "modelos_selecionados": selected_models,
        "intervalo": "conformal de 90%, calibrado com resíduos da janela temporal 2017-2018",
        "faixas_risco": RISK_LABELS,
        "limites_faixas_probabilidade": [0.25, 0.50, 0.75],
        "contagem": "proxy derivada da taxa prevista e da exposição populacional de dois anos",
        "avaliacao": "não disponível até a publicação dos valores observados de 2025 e 2026",
        "fontes": _source_metadata(raw_dir),
    }
    (output_dir / "config_futuro.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    run_future_forecast()


if __name__ == "__main__":
    main()
