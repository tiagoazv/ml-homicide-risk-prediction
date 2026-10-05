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

TEMPORAL_FEATURE_COLUMNS = [
    "taxa_mediana_3_anos",
    "taxa_min_3_anos",
    "taxa_max_3_anos",
    "amplitude_taxa_3_anos",
    "tendencia_taxa_3_anos",
    "tendencia_log_3_anos",
    "aumentos_3_anos",
]

FEATURE_COLUMNS_WITH_3_YEAR_STATS = FEATURE_COLUMNS + TEMPORAL_FEATURE_COLUMNS

# Conjunto base de features. O modelo utiliza o histórico recente do Atlas e
# a exposição populacional na origem da previsão.
FEATURE_COLUMNS_BASE = [
    "ano",
    "taxa_homicidios",
    "taxa_lag_1",
    "taxa_lag_2",
    "taxa_media_3_anos",
    "taxa_std_3_anos",
    "taxa_mediana_3_anos",
    "tendencia_log_3_anos",
    "variacao_log_100_atual",
    "log_populacao",
]

FIVE_YEAR_FEATURE_COLUMNS = [
    "taxa_media_5_anos",
    "taxa_std_5_anos",
    "taxa_mediana_5_anos",
    "amplitude_taxa_5_anos",
    "tendencia_taxa_5_anos",
    "tendencia_log_5_anos",
    "aumentos_5_anos",
]

# Conjunto final: features recentes mais estatísticas calculadas em cinco anos.
FEATURE_COLUMNS_FINAL = FEATURE_COLUMNS_BASE + FIVE_YEAR_FEATURE_COLUMNS

TARGET_LOG_COLUMN = "target_variacao_log_100"
TARGET_PERCENT_COLUMN = "target_variacao_percentual"
TARGET_RATE_LOG_COLUMN = "target_log_taxa_100"
COUNT_TARGET_COLUMN = "homicidios_proxy_futuro"
OFFICIAL_COUNT_TARGET_COLUMN = "homicidios_sim_futuro"
SMOOTHED_RATE_TARGET_COLUMN = "taxa_futura_media_3_anos"
SMOOTHED_RATE_LOG_TARGET_COLUMN = "target_log_taxa_futura_media_3_anos_100"
SMOOTHED_RATE_PERCENT_TARGET_COLUMN = "target_variacao_media_3_anos_percentual"


def smoothed_target_columns(horizon: int = 3) -> tuple[str, str, str]:
    """Retorna as colunas do alvo suavizado para um horizonte válido."""

    if horizon < 1:
        raise ValueError("horizon deve ser um inteiro positivo.")
    return (
        f"taxa_futura_media_{horizon}_anos",
        f"target_log_taxa_futura_media_{horizon}_anos_100",
        f"target_variacao_media_{horizon}_anos_percentual",
    )


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
    frame: pd.DataFrame,
    population: pd.DataFrame | None = None,
    sim_counts: pd.DataFrame | None = None,
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
    data["taxa_lag_3"] = grouped["taxa_homicidios"].shift(3)
    data["taxa_lag_4"] = grouped["taxa_homicidios"].shift(4)
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
    historico_3_anos = data[["taxa_lag_2", "taxa_lag_1", "taxa_homicidios"]]
    data["taxa_mediana_3_anos"] = np.where(
        tres_anos_consecutivos,
        historico_3_anos.median(axis=1),
        np.nan,
    )
    data["taxa_min_3_anos"] = np.where(
        tres_anos_consecutivos,
        historico_3_anos.min(axis=1),
        np.nan,
    )
    data["taxa_max_3_anos"] = np.where(
        tres_anos_consecutivos,
        historico_3_anos.max(axis=1),
        np.nan,
    )
    data["amplitude_taxa_3_anos"] = np.where(
        tres_anos_consecutivos,
        data["taxa_max_3_anos"] - data["taxa_min_3_anos"],
        np.nan,
    )
    data["tendencia_taxa_3_anos"] = np.where(
        tres_anos_consecutivos,
        (data["taxa_homicidios"] - data["taxa_lag_2"]) / 2,
        np.nan,
    )
    data["tendencia_log_3_anos"] = np.where(
        tres_anos_consecutivos,
        (
            np.log1p(data["taxa_homicidios"])
            - np.log1p(data["taxa_lag_2"])
        )
        / 2
        * 100,
        np.nan,
    )
    data["aumentos_3_anos"] = np.where(
        tres_anos_consecutivos,
        data["taxa_lag_1"].gt(data["taxa_lag_2"]).astype(int)
        + data["taxa_homicidios"].gt(data["taxa_lag_1"]).astype(int),
        np.nan,
    )
    data["ano_lag_3"] = grouped["ano"].shift(3)
    data["ano_lag_4"] = grouped["ano"].shift(4)
    intervalo_terceiro = data["ano_lag_2"].sub(data["ano_lag_3"])
    intervalo_quarto = data["ano_lag_3"].sub(data["ano_lag_4"])
    cinco_anos_consecutivos = (
        tres_anos_consecutivos
        & intervalo_terceiro.eq(1).fillna(False)
        & intervalo_quarto.eq(1).fillna(False)
    )
    historico_5_anos = data[
        [
            "taxa_lag_4",
            "taxa_lag_3",
            "taxa_lag_2",
            "taxa_lag_1",
            "taxa_homicidios",
        ]
    ]
    data["taxa_media_5_anos"] = np.where(
        cinco_anos_consecutivos, historico_5_anos.mean(axis=1), np.nan
    )
    data["taxa_std_5_anos"] = np.where(
        cinco_anos_consecutivos,
        historico_5_anos.std(axis=1, ddof=0),
        np.nan,
    )
    data["taxa_mediana_5_anos"] = np.where(
        cinco_anos_consecutivos, historico_5_anos.median(axis=1), np.nan
    )
    data["amplitude_taxa_5_anos"] = np.where(
        cinco_anos_consecutivos,
        historico_5_anos.max(axis=1) - historico_5_anos.min(axis=1),
        np.nan,
    )
    data["tendencia_taxa_5_anos"] = np.where(
        cinco_anos_consecutivos,
        (data["taxa_homicidios"] - data["taxa_lag_4"]) / 4,
        np.nan,
    )
    data["tendencia_log_5_anos"] = np.where(
        cinco_anos_consecutivos,
        (
            np.log1p(data["taxa_homicidios"])
            - np.log1p(data["taxa_lag_4"])
        )
        / 4
        * 100,
        np.nan,
    )
    data["aumentos_5_anos"] = np.where(
        cinco_anos_consecutivos,
        data["taxa_lag_3"].gt(data["taxa_lag_4"]).astype(int)
        + data["taxa_lag_2"].gt(data["taxa_lag_3"]).astype(int)
        + data["taxa_lag_1"].gt(data["taxa_lag_2"]).astype(int)
        + data["taxa_homicidios"].gt(data["taxa_lag_1"]).astype(int),
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
    data[TARGET_RATE_LOG_COLUMN] = np.where(
        um_ano_futuro,
        100 * np.log1p(data["taxa_futuro"]),
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
        result["homicidios_proxy_atual"] = (
            result["taxa_homicidios"] * result["populacao"] / 100_000
        )
        result[COUNT_TARGET_COLUMN] = (
            result["taxa_futuro"] * result["populacao"] / 100_000
        )
    if sim_counts is not None:
        required_sim = {"municipio_codigo", "ano", "homicidios_sim"}
        missing_sim = required_sim.difference(sim_counts.columns)
        if missing_sim:
            raise ValueError(
                f"Colunas ausentes na contagem oficial: {sorted(missing_sim)}"
            )
        official = sim_counts[
            ["municipio_codigo", "ano", "homicidios_sim"]
        ].copy()
        official["municipio_codigo"] = official["municipio_codigo"].astype("string")
        official["ano"] = pd.to_numeric(official["ano"], errors="raise").astype(int)
        if official.duplicated(["municipio_codigo", "ano"]).any():
            raise ValueError("A contagem oficial contém município-ano duplicado.")

        current = official.rename(
            columns={"ano": "ano", "homicidios_sim": "homicidios_sim_atual"}
        )
        future = official.rename(
            columns={"ano": "ano_futuro", "homicidios_sim": OFFICIAL_COUNT_TARGET_COLUMN}
        )
        result = result.merge(
            current, on=["municipio_codigo", "ano"], how="left", validate="one_to_one"
        )
        result = result.merge(
            future,
            on=["municipio_codigo", "ano_futuro"],
            how="left",
            validate="one_to_one",
        )
        # O recorte oficial pode cobrir somente parte da série histórica; anos
        # fora da cobertura são removidos, nunca tratados como zero.
        result = result.dropna(
            subset=["homicidios_sim_atual", OFFICIAL_COUNT_TARGET_COLUMN]
        ).copy()
        result["homicidios_sim_atual"] = result["homicidios_sim_atual"].astype(int)
        result[OFFICIAL_COUNT_TARGET_COLUMN] = result[
            OFFICIAL_COUNT_TARGET_COLUMN
        ].astype(int)
        result["taxa_homicidios_sim_atual"] = (
            100_000 * result["homicidios_sim_atual"] / result["populacao"]
        )
        result["taxa_homicidios_sim_futuro"] = (
            100_000 * result[OFFICIAL_COUNT_TARGET_COLUMN] / result["populacao"]
        )
    return result.reset_index(drop=True)


def get_feature_columns(data: pd.DataFrame) -> list[str]:
    """Retorna as features disponíveis, permitindo testes sem população."""

    columns = [column for column in FEATURE_COLUMNS if column in data.columns]
    if len(columns) < len(FEATURE_COLUMNS) and "log_populacao" in data.columns:
        raise ValueError("A feature de população está presente, mas não foi reconhecida.")
    return columns


def build_smoothed_model_dataset(
    frame: pd.DataFrame,
    population: pd.DataFrame | None = None,
    horizon: int = 3,
    history_years: int = 3,
) -> pd.DataFrame:
    """Cria um painel com alvo médio dos anos futuros consecutivos.

    Para uma linha de origem ``t``, o alvo é a média das taxas de ``t+1`` até
    ``t+horizon``. Todos os anos precisam existir consecutivamente;
    assim, uma ausência no Atlas não é transformada artificialmente em zero.
    A construção reutiliza as features históricas validadas e acrescenta
    somente o horizonte futuro suavizado. O experimento final exige cinco anos
    históricos para habilitar as features de tendência ampliada.
    """

    if history_years not in (3, 5):
        raise ValueError("history_years deve ser 3 ou 5.")

    target_column, log_target_column, percent_target_column = smoothed_target_columns(
        horizon
    )

    base = build_model_dataset(frame, population=population)
    data = frame.copy().sort_values(["municipio_codigo", "ano"])
    grouped = data.groupby("municipio_codigo", sort=False)
    future_columns = []
    for offset in range(1, horizon + 1):
        rate_column = f"taxa_futura_{offset}"
        year_column = f"ano_futuro_{offset}"
        data[rate_column] = grouped["taxa_homicidios"].shift(-offset)
        data[year_column] = grouped["ano"].shift(-offset)
        future_columns.extend([rate_column, year_column])

    valid = data[future_columns].notna().all(axis=1)
    valid &= data["ano_futuro_1"].sub(data["ano"]).eq(1)
    for offset in range(2, horizon + 1):
        valid &= data[f"ano_futuro_{offset}"].sub(data["ano"]).eq(offset)
    future = data.loc[
        valid,
        ["municipio_codigo", "ano", *future_columns],
    ].copy()
    future[target_column] = future[
        [f"taxa_futura_{offset}" for offset in range(1, horizon + 1)]
    ].mean(axis=1)
    future["ano_futuro_inicio"] = future["ano_futuro_1"].astype(int)
    future["ano_futuro_final"] = future[f"ano_futuro_{horizon}"].astype(int)

    result = base.merge(
        future[
            [
                "municipio_codigo",
                "ano",
                target_column,
                "ano_futuro_inicio",
                "ano_futuro_final",
            ]
        ],
        on=["municipio_codigo", "ano"],
        how="inner",
        validate="one_to_one",
    )
    if history_years == 5:
        result = result.dropna(subset=FIVE_YEAR_FEATURE_COLUMNS).copy()
    result[log_target_column] = 100 * np.log1p(result[target_column])
    current = result["taxa_media_3_anos"]
    result[percent_target_column] = np.where(
        current.gt(0),
        (result[target_column] - current) / current * 100,
        np.nan,
    )
    # ano_futuro identifies the final year of the forecast horizon.
    result["ano_futuro"] = result["ano_futuro_final"]
    return result.reset_index(drop=True)


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
