"""Carregamento e integração da população municipal do IBGE."""

import json

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


def load_sidra_population_json(path: str | Path) -> pd.DataFrame:
    """Converte uma resposta JSON municipal do SIDRA para o contrato local."""

    records = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("A resposta SIDRA precisa ser uma lista não vazia.")
    data = pd.DataFrame(records)
    required = {"D1C", "D3C", "V"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Campos ausentes na resposta SIDRA: {sorted(missing)}")

    result = pd.DataFrame(
        {
            "municipio_codigo": data["D1C"].astype("string").str.strip().str.zfill(7),
            "ano": pd.to_numeric(data["D3C"], errors="coerce"),
            "populacao": pd.to_numeric(data["V"], errors="coerce"),
            "fonte_tabela": Path(path).stem,
        }
    )
    result = result.dropna(subset=["municipio_codigo", "ano", "populacao"])
    if result.empty:
        raise ValueError("A resposta SIDRA não contém observações municipais válidas.")
    result["ano"] = result["ano"].astype(int)
    result["populacao"] = result["populacao"].astype(int)
    if result["municipio_codigo"].str.fullmatch(r"\d{7}").eq(False).any():
        raise ValueError("Há códigos municipais inválidos na resposta SIDRA.")
    if result["populacao"].le(0).any():
        raise ValueError("A resposta SIDRA contém populações não positivas.")
    if result.duplicated(["municipio_codigo", "ano"]).any():
        raise ValueError("A resposta SIDRA contém município-ano duplicado.")
    return result.sort_values(["municipio_codigo", "ano"]).reset_index(drop=True)


def combine_population_sources(*sources: pd.DataFrame) -> pd.DataFrame:
    """Une fontes populacionais sem aceitar conflito para município-ano."""

    if not sources:
        raise ValueError("Informe ao menos uma fonte populacional.")
    combined = pd.concat(sources, ignore_index=True)
    required = set(POPULATION_COLUMNS).difference(combined.columns)
    if required:
        raise ValueError(f"Colunas ausentes na população: {sorted(required)}")
    conflicts = combined.groupby(["municipio_codigo", "ano"])["populacao"].nunique()
    if conflicts.gt(1).any():
        raise ValueError("Fontes populacionais discordam para município-ano.")
    return (
        combined.drop_duplicates(["municipio_codigo", "ano"], keep="last")
        .sort_values(["municipio_codigo", "ano"])
        .reset_index(drop=True)
    )


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
    history = population.rename(columns={"ano": "populacao_ano_referencia"})
    data["municipio_codigo"] = data.municipio_codigo.astype("string")
    history["municipio_codigo"] = history.municipio_codigo.astype("string")
    data["ano"] = pd.to_numeric(data["ano"], errors="raise").astype("int64")
    history["populacao_ano_referencia"] = pd.to_numeric(
        history["populacao_ano_referencia"], errors="raise"
    ).astype("int64")
    join_year = "populacao_disponivel_desde" if "populacao_disponivel_desde" in history else "populacao_ano_referencia"
    data = pd.merge_asof(
        data.sort_values("ano"), history.sort_values(join_year),
        left_on="ano", right_on=join_year,
        by="municipio_codigo", direction="backward",
    )
    data["populacao_observada"] = data.ano.eq(data.populacao_ano_referencia)

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
