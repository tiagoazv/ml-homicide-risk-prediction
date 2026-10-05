"""Leitura auditável das séries oficiais e integração retrospectiva municipal."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

SOCIAL_COLUMNS = ["gini", "idhm", "pind", "analfabetismo_15_mais"]
SERIES = dict(zip(SOCIAL_COLUMNS, ["ADH_GINI", "ADH_IDHM", "ADH_PIND", "ADH_T_ANALF15M"]))


def municipality_lookup(population, registry_path):
    """O prefixo de seis dígitos é resolvido pela população, sem inventar DV."""
    codes = population[["municipio_codigo"]].drop_duplicates().copy()
    codes["codigo_atlas"] = codes.municipio_codigo.str[:6]
    if not codes.municipio_codigo.str.fullmatch(r"\d{7}").all() or codes.codigo_atlas.duplicated().any():
        raise ValueError("Códigos municipais inválidos ou prefixos ambíguos.")
    registry = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    names = pd.DataFrame([
        {"municipio_codigo": str(row["id"]), "municipio": row["nome"],
         "uf_codigo": str(row["id"])[:2]} for row in registry
    ])
    result = codes.merge(names, on="municipio_codigo", how="left", validate="one_to_one")
    if result.municipio.isna().any():
        raise ValueError("Há códigos populacionais sem nome no cadastro do IBGE.")
    return result


def load_atlas_rates(path, lookup):
    data = pd.read_csv(path, encoding="utf-8-sig", dtype={"Região ID": "string"})
    data = data.rename(columns={"Região ID": "codigo_atlas", "Valor": "taxa_homicidios"})
    data["ano"] = pd.to_datetime(data["Período"], utc=True, errors="raise").dt.year
    data["taxa_homicidios"] = pd.to_numeric(data.taxa_homicidios, errors="raise")
    if not data.codigo_atlas.str.fullmatch(r"\d{6}").all():
        raise ValueError("Código Atlas deve ter seis dígitos.")
    if not np.isfinite(data.taxa_homicidios).all() or data.taxa_homicidios.lt(0).any():
        raise ValueError("Taxa inválida.")
    if data.duplicated(["codigo_atlas", "ano"]).any():
        raise ValueError("Taxa duplicada para município-ano.")
    result = data.merge(lookup, on="codigo_atlas", how="left", validate="many_to_one")
    if result.municipio_codigo.isna().any():
        raise ValueError("Código Atlas sem correspondência na população.")
    return result.drop(columns="Período").sort_values(["municipio_codigo", "ano"]).reset_index(drop=True)


def load_social(path, series, column):
    data = pd.read_csv(path, encoding="utf-8-sig", dtype={"TERCODIGO": "string"})
    data = data.loc[data.NIVNOME.eq("Municípios") & data.SERCODIGO.eq(series)].copy()
    if data.empty or not data.TERCODIGO.str.fullmatch(r"\d{7}").all():
        raise ValueError(f"Série municipal inválida: {series}")
    data["ano_referencia"] = pd.to_numeric(data.VALDATA.str[:4], errors="raise").astype(int)
    data[column] = pd.to_numeric(data.VALVALOR, errors="raise")
    upper = 1 if column in ("gini", "idhm") else 100
    if not data[column].between(0, upper).all():
        raise ValueError(f"Indicador fora da escala: {series}")
    data = data.rename(columns={"TERCODIGO": "municipio_codigo"})
    if data.duplicated(["municipio_codigo", "ano_referencia"]).any():
        raise ValueError(f"Indicador duplicado: {series}")
    return data[["municipio_codigo", "ano_referencia", column]]


def attach_social(panel, raw_dir, release_lag=3, publication_years=None):
    """Último censo anterior; lag é hipótese explícita, não data de publicação."""
    if release_lag < 0:
        raise ValueError("Defasagem não pode ser negativa.")
    result = panel.copy()
    result["municipio_codigo"] = result.municipio_codigo.astype("string")
    for column, series in SERIES.items():
        data = load_social(Path(raw_dir) / f"{series}.csv", series, column)
        ref = f"{column}_ano_referencia"
        available = f"{column}_ano_disponibilidade_assumido"
        data = data.rename(columns={"ano_referencia": ref})
        data[available] = data[ref] + release_lag
        if publication_years is not None:
            data[available] = data[ref].map(publication_years)
            if data[available].isna().any():
                raise ValueError("Referência sem ano de publicação documentado.")
            data[available] = data[available].astype(int)
        result = pd.merge_asof(
            result.sort_values("ano"), data.sort_values([available, ref, "municipio_codigo"]),
            left_on="ano", right_on=available, by="municipio_codigo", direction="backward",
        )
        result[f"{column}_idade_anos"] = result.ano - result[ref]
    return result.sort_values(["municipio_codigo", "ano"]).reset_index(drop=True)
