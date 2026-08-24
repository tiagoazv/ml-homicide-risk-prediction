"""Preparação da base municipal para o MVP analítico."""

from pathlib import Path

import numpy as np
import pandas as pd


COLUMN_MAP = {
    "cod": "municipio_codigo",
    "nome": "municipio",
    "período": "ano",
    "valor": "taxa_homicidios",
}

FEATURE_COLUMNS = [
    "ano",
    "taxa_homicidios",
    "taxa_lag_1",
    "taxa_lag_2",
    "taxa_media_3_anos",
    "taxa_std_3_anos",
    "variacao_log_100_atual",
    "log_populacao",
]

TARGET_LOG_COLUMN = "target_variacao_log_100"
TARGET_PERCENT_COLUMN = "target_variacao_percentual"


def load_homicide_rates(path: str | Path) -> pd.DataFrame:
    """Carrega e padroniza o CSV de taxas do Atlas."""

    frame = pd.read_csv(path, sep=";", encoding="utf-8")
    renamed = frame.rename(columns=COLUMN_MAP).copy()
    required = set(COLUMN_MAP.values())
    missing = required.difference(renamed.columns)
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {sorted(missing)}")

    renamed["municipio_codigo"] = (
        renamed["municipio_codigo"].astype("string").str.strip().str.zfill(7)
    )
    renamed["municipio"] = renamed["municipio"].astype("string").str.strip()
    renamed["ano"] = pd.to_numeric(renamed["ano"], errors="coerce")
    renamed["taxa_homicidios"] = pd.to_numeric(
        renamed["taxa_homicidios"], errors="coerce"
    )

    if renamed[["municipio_codigo", "ano", "taxa_homicidios"]].isna().any().any():
        raise ValueError("A base contém valores ausentes em campos essenciais.")
    if renamed["taxa_homicidios"].lt(0).any():
        raise ValueError("A base contém taxas negativas.")
    if renamed.duplicated(["municipio_codigo", "ano"]).any():
        raise ValueError("A base contém mais de uma observação para município-ano.")

    renamed["ano"] = renamed["ano"].astype(int)
    return renamed.sort_values(["municipio_codigo", "ano"]).reset_index(drop=True)


def build_model_dataset(
    frame: pd.DataFrame, population: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Cria features históricas e o alvo do ano seguinte.

    O alvo primário é a diferença logarítmica multiplicada por 100. A conversão
    para uma variação percentual estabilizada é feita por ``log_change_to_percent``.
    Linhas sem três anos consecutivos de histórico ou sem o ano seguinte são
    removidas para impedir comparações temporais inválidas.
    """

    data = frame.copy().sort_values(["municipio_codigo", "ano"])
    grouped = data.groupby("municipio_codigo", sort=False)

    data["taxa_lag_1"] = grouped["taxa_homicidios"].shift(1)
    data["ano_lag_1"] = grouped["ano"].shift(1)
    data["taxa_lag_2"] = grouped["taxa_homicidios"].shift(2)
    data["ano_lag_2"] = grouped["ano"].shift(2)
    data["taxa_futuro"] = grouped["taxa_homicidios"].shift(-1)
    data["ano_futuro"] = grouped["ano"].shift(-1)

    intervalo_anterior = data["ano"].sub(data["ano_lag_1"])
    intervalo_mais_antigo = data["ano_lag_1"].sub(data["ano_lag_2"])
    intervalo_futuro = data["ano_futuro"].sub(data["ano"])

    um_ano_anterior = intervalo_anterior.eq(1).fillna(False)
    tres_anos_consecutivos = (
        um_ano_anterior & intervalo_mais_antigo.eq(1).fillna(False)
    )
    um_ano_futuro = intervalo_futuro.eq(1).fillna(False)

    data["taxa_media_3_anos"] = np.where(
        tres_anos_consecutivos,
        data[["taxa_lag_2", "taxa_lag_1", "taxa_homicidios"]].mean(axis=1),
        np.nan,
    )
    data["taxa_std_3_anos"] = np.where(
        tres_anos_consecutivos,
        data[["taxa_lag_2", "taxa_lag_1", "taxa_homicidios"]].std(
            axis=1, ddof=0
        ),
        np.nan,
    )
    data["variacao_log_100_atual"] = np.where(
        um_ano_anterior,
        100
        * (
            np.log1p(data["taxa_homicidios"])
            - np.log1p(data["taxa_lag_1"])
        ),
        np.nan,
    )

    data[TARGET_LOG_COLUMN] = np.where(
        um_ano_futuro,
        100
        * (
            np.log1p(data["taxa_futuro"])
            - np.log1p(data["taxa_homicidios"])
        ),
        np.nan,
    )
    data[TARGET_PERCENT_COLUMN] = np.where(
        um_ano_futuro & data["taxa_homicidios"].gt(0),
        (data["taxa_futuro"] - data["taxa_homicidios"])
        / data["taxa_homicidios"]
        * 100,
        np.nan,
    )

    valid = tres_anos_consecutivos & um_ano_futuro
    result = data.loc[valid].copy()
    result["ano"] = result["ano"].astype(int)
    result["ano_futuro"] = result["ano_futuro"].astype(int)
    if population is not None:
        from .population import attach_population_features

        population_years = population["ano"].astype(int)
        result = result[
            result["ano"].between(population_years.min(), population_years.max())
        ].copy()
        result = attach_population_features(result, population)
    return result.reset_index(drop=True)


def get_feature_columns(data: pd.DataFrame) -> list[str]:
    """Retorna as features disponíveis, permitindo testes sem população."""

    columns = [column for column in FEATURE_COLUMNS if column in data.columns]
    if len(columns) < len(FEATURE_COLUMNS) and "log_populacao" in data.columns:
        raise ValueError("A feature de população está presente, mas não foi reconhecida.")
    return columns


def split_by_year(
    data: pd.DataFrame, train_end: int = 2016, validation_end: int = 2019
) -> dict[str, pd.DataFrame]:
    """Divide o painel por ano de origem, sem embaralhamento."""

    train = data[data["ano"] <= train_end].copy()
    validation = data[
        data["ano"].gt(train_end) & data["ano"].le(validation_end)
    ].copy()
    test = data[data["ano"].gt(validation_end)].copy()

    if train.empty or validation.empty or test.empty:
        raise ValueError("A divisão temporal produziu um conjunto vazio.")
    if train["ano"].max() >= validation["ano"].min():
        raise ValueError("Sobreposição entre treino e validação.")
    if validation["ano"].max() >= test["ano"].min():
        raise ValueError("Sobreposição entre validação e teste.")

    return {"train": train, "validation": validation, "test": test}


def log_change_to_percent(values: pd.Series | np.ndarray) -> np.ndarray:
    """Converte a variação logarítmica em percentual estabilizado."""

    return np.expm1(np.asarray(values) / 100) * 100
