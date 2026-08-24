"""Carregamento e integração da população municipal do IBGE."""

from pathlib import Path

import numpy as np
import pandas as pd


POPULATION_COLUMNS = ["municipio_codigo", "ano", "populacao", "fonte_tabela"]
POPULATION_SIZE_LABELS = [
    "ate_10_mil",
    "10_mil_a_50_mil",
    "50_mil_a_200_mil",
    "200_mil_a_500_mil",
    "acima_500_mil",
]


def filter_by_population(
    frame: pd.DataFrame, minimum: int = 50_000
) -> pd.DataFrame:
    """Seleciona a coorte por população no ano de origem."""

    if minimum <= 0:
        raise ValueError("O limite populacional deve ser positivo.")
    if "populacao" not in frame.columns:
        raise ValueError("A base precisa conter a coluna populacao.")
    selected = frame.loc[frame["populacao"].ge(minimum)].copy()
    if selected.empty:
        raise ValueError("O limite populacional não selecionou observações.")
    return selected.reset_index(drop=True)


def load_population(path: str | Path) -> pd.DataFrame:
    """Carrega a população observada e valida a chave município-ano."""

    population = pd.read_csv(
        path,
        dtype={"municipio_codigo": "string", "fonte_tabela": "string"},
    )
    missing = set(POPULATION_COLUMNS).difference(population.columns)
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {sorted(missing)}")

    population["municipio_codigo"] = (
        population["municipio_codigo"].str.strip().str.zfill(7)
    )
    population["ano"] = pd.to_numeric(population["ano"], errors="coerce")
    population["populacao"] = pd.to_numeric(
        population["populacao"], errors="coerce"
    )

    if population[["municipio_codigo", "ano", "populacao"]].isna().any().any():
        raise ValueError("A população contém valores ausentes em campos essenciais.")
    if population["populacao"].le(0).any():
        raise ValueError("A população contém valores não positivos.")
    if population.duplicated(["municipio_codigo", "ano"]).any():
        raise ValueError("A população contém mais de uma observação para município-ano.")

    population["ano"] = population["ano"].astype(int)
    population["populacao"] = population["populacao"].astype(int)
    return population.sort_values(["municipio_codigo", "ano"]).reset_index(drop=True)


def attach_population_features(
    frame: pd.DataFrame, population: pd.DataFrame
) -> pd.DataFrame:
    """Anexa população ao painel e marca valores preenchidos por retenção.

    A população é uma feature contemporânea ao ano de origem. Quando a fonte
    não publica um valor para um município-ano, usa-se a última observação
    anterior disponível, sem interpolar com um ano futuro. Linhas anteriores
    à primeira observação populacional do município são descartadas.
    """

    data = frame.copy()
    data["_ordem_original"] = np.arange(len(data))
    data = data.merge(
        population,
        on=["municipio_codigo", "ano"],
        how="left",
        validate="many_to_one",
    ).sort_values(["municipio_codigo", "ano"])
    data["populacao_observada"] = data["populacao"].notna()
    data["populacao"] = data.groupby("municipio_codigo", sort=False)[
        "populacao"
    ].ffill()

    data = data.loc[data["populacao"].notna()].copy()
    if data.empty:
        raise ValueError("Não foi possível obter população para nenhuma observação.")

    data["populacao"] = data["populacao"].astype(int)
    data["populacao_fonte"] = np.where(
        data["populacao_observada"],
        "observada",
        "ultima_observacao_anterior",
    )
    data["log_populacao"] = np.log1p(data["populacao"])
    data["porte_populacional"] = pd.cut(
        data["populacao"],
        bins=[0, 10_000, 50_000, 200_000, 500_000, np.inf],
        labels=POPULATION_SIZE_LABELS,
        right=True,
    ).astype("string")

    return data.sort_values("_ordem_original").drop(columns="_ordem_original")
