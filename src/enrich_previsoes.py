"""Enriquece as previsões para a representação cartográfica."""

import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .models_final import _authorized_features
from .prepare_data import FEATURE_COLUMNS_FINAL, smoothed_target_columns, split_by_year


RISK_BINS = [-np.inf, 0.25, 0.50, 0.75, np.inf]
RISK_LABELS = ["baixo", "moderado", "alto", "muito_alto"]


def _conformal_quantile(residuals: np.ndarray, confidence: float) -> float:
    if not 0 < confidence < 1:
        raise ValueError("confidence deve estar entre zero e um.")
    values = np.asarray(residuals, dtype=float)
    if values.size == 0:
        raise ValueError("É necessário ter resíduos de validação.")
    level = min(1.0, math.ceil((len(values) + 1) * confidence) / len(values))
    try:
        return float(np.quantile(values, level, method="higher"))
    except TypeError:
        return float(np.quantile(values, level, interpolation="higher"))


def _interval_for_window(
    panel: pd.DataFrame,
    selected_rate: pd.DataFrame,
    model_root: Path,
    cohort: str,
    window: str,
    train_end: int,
    validation_end: int,
    test_end: int,
    horizon: int,
    confidence: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    target_column, log_target_column, _ = smoothed_target_columns(horizon)
    current = panel[
        panel.ano.le(test_end)
        & (
            panel.populacao.ge(50_000)
            if cohort == "principal_50mil"
            else panel.municipio_codigo.notna()
        )
    ].copy()
    splits = split_by_year(
        current, train_end=train_end, validation_end=validation_end
    )
    selected = selected_rate[
        selected_rate.coorte.eq(cohort) & selected_rate.janela.eq(window)
    ].copy()
    if selected.empty:
        raise ValueError(f"Não há previsões selecionadas para {cohort}/{window}.")
    model_name = str(selected.modelo.iloc[0])
    model_path = model_root / "taxa_suavizada" / cohort / window / f"{model_name}.joblib"
    model = joblib.load(model_path)
    features = _authorized_features(panel, FEATURE_COLUMNS_FINAL)
    validation = splits["validation"]
    validation_prediction = model.predict(validation[features])
    residuals = np.abs(
        validation[log_target_column].to_numpy() - validation_prediction
    )
    quantile = _conformal_quantile(residuals, confidence)
    predicted_log_column = f"previsao_log_taxa_media_{horizon}_anos_100"
    lower_log = selected[predicted_log_column] - quantile
    upper_log = selected[predicted_log_column] + quantile
    selected[f"intervalo_inferior_taxa_media_{horizon}_anos"] = np.maximum(
        0, np.expm1(lower_log / 100)
    )
    selected[f"intervalo_superior_taxa_media_{horizon}_anos"] = np.maximum(
        0, np.expm1(upper_log / 100)
    )
    selected["quantil_conformal_log_100"] = quantile
    actual_log = selected[f"alvo_log_taxa_media_{horizon}_anos_100"]
    covered = actual_log.between(
        selected[predicted_log_column] - quantile,
        selected[predicted_log_column] + quantile,
    )
    interval_metrics = {
        "coorte": cohort,
        "janela": window,
        "modelo": model_name,
        "confianca_nominal": confidence,
        "quantil_conformal_log_100": quantile,
        "n_validacao": len(validation),
        "n_teste": len(selected),
        "cobertura_teste": float(covered.mean()),
        "largura_media_taxa": float(
            (
                selected[f"intervalo_superior_taxa_media_{horizon}_anos"]
                - selected[f"intervalo_inferior_taxa_media_{horizon}_anos"]
            ).mean()
        ),
    }
    return selected, interval_metrics


def _add_rank_and_risk_band(
    data: pd.DataFrame, predicted_column: str
) -> pd.DataFrame:
    result = data.copy()
    group = ["coorte", "janela", "ano"]
    result["ranking_taxa_futura"] = result.groupby(group)[
        predicted_column
    ].rank(method="min", ascending=False).astype("Int64")
    counts = result.groupby(group)[predicted_column].transform("count")
    result["percentil_ranking_taxa"] = np.where(
        counts.gt(1),
        1 - (result["ranking_taxa_futura"] - 1) / (counts - 1),
        1.0,
    )
    if "probabilidade_alto_risco_calibrada" in result:
        probability = result["probabilidade_alto_risco_calibrada"]
        result["faixa_risco"] = pd.cut(
            probability,
            bins=RISK_BINS,
            labels=RISK_LABELS,
            include_lowest=True,
        ).astype("string")
    return result


def enrich_predictions(
    output_dir: Path,
    model_root: Path,
    panel: pd.DataFrame,
    windows: list[tuple[int, int, int]],
    horizon: int = 2,
    confidence: float = 0.90,
) -> pd.DataFrame:
    """Adiciona intervalos, ranking e faixas sem alterar o alvo ou a seleção."""

    rate_path = output_dir / "previsoes_taxa_suavizada_mapa.csv"
    map_path = output_dir / "previsoes_mapa.csv"
    selected_rate = pd.read_csv(rate_path, dtype={"municipio_codigo": "string"})
    enriched_parts: list[pd.DataFrame] = []
    interval_rows: list[dict[str, object]] = []
    for train_end, validation_end, test_end in windows:
        window = f"{validation_end + 1}-{test_end}"
        for cohort in ("principal_50mil", "todos"):
            part, metrics = _interval_for_window(
                panel,
                selected_rate,
                model_root,
                cohort,
                window,
                train_end,
                validation_end,
                test_end,
                horizon,
                confidence,
            )
            enriched_parts.append(part)
            interval_rows.append(metrics)
    enriched_rate = pd.concat(enriched_parts, ignore_index=True)
    predicted_column = f"taxa_media_{horizon}_anos_prevista"
    enriched_rate = _add_rank_and_risk_band(enriched_rate, predicted_column)
    if "faixa_risco" not in enriched_rate:
        enriched_rate["faixa_risco"] = pd.Series(
            pd.NA, index=enriched_rate.index, dtype="string"
        )
    enriched_rate.to_csv(rate_path, index=False)

    map_data = pd.read_csv(map_path, dtype={"municipio_codigo": "string"})
    key = ["coorte", "experimento", "janela", "municipio_codigo", "ano"]
    added_columns = [
        "quantil_conformal_log_100",
        f"intervalo_inferior_taxa_media_{horizon}_anos",
        f"intervalo_superior_taxa_media_{horizon}_anos",
        "ranking_taxa_futura",
        "percentil_ranking_taxa",
        "faixa_risco",
    ]
    map_data = map_data.drop(columns=[column for column in added_columns if column in map_data])
    map_data = map_data.merge(
        enriched_rate[key + added_columns],
        on=key,
        how="left",
        validate="one_to_one",
    )
    map_data = _add_rank_and_risk_band(map_data, predicted_column)
    map_data.to_csv(map_path, index=False)
    interval_frame = pd.DataFrame(interval_rows)
    interval_frame.to_csv(output_dir / "intervalos_conformais_v11.csv", index=False)
    return map_data
