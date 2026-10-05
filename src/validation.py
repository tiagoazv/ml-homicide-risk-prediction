"""Atualização populacional e comparação temporal em janelas sem teste sobreposto."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .population import load_population


def updated_population(raw):
    base = load_population(raw / "populacao_municipal_ibge_2001_2021.csv")
    base["populacao_disponivel_desde"] = base.ano
    additions = []
    for year, table, available in [(2022, "4714", 2023), (2024, "6579", 2024)]:
        rows = json.loads((raw / f"sidra_populacao_{year}.json").read_text())
        for row in rows[1:]:
            if row["NC"] != "6" or row["D3C"] != str(year):
                raise ValueError("Resposta SIDRA com território/período inesperado.")
            if row["V"] in {"...", "..", "X"}:
                # Marcadores de indisponibilidade/sigilo não são população zero.
                continue
            value = pd.to_numeric(row["V"], errors="raise")
            if value <= 0 or not np.isfinite(value):
                raise ValueError("População SIDRA inválida.")
            additions.append({"municipio_codigo": row["D1C"], "ano": year,
                "populacao": int(value), "fonte_tabela": table, "populacao_disponivel_desde": available})
    result = pd.concat([base, pd.DataFrame(additions)], ignore_index=True)
    result["municipio_codigo"] = result.municipio_codigo.astype("string")
    if result.duplicated(["municipio_codigo", "ano"]).any():
        raise ValueError("População duplicada.")
    return result.sort_values(["municipio_codigo", "ano"])


def cluster_bootstrap(predictions, iterations=1000, seed=42):
    """IC exploratório da diferença MAE social − histórico; reamostra cidades."""
    forest = predictions[predictions.modelo.eq("random_forest") & predictions.divisao.eq("test")]
    rows = []
    for cohort, group in forest.groupby("coorte"):
        key = ["municipio_codigo", "ano", "janela"]
        historical = group[group.experimento.eq("historico")].copy()
        social = group[group.experimento.eq("social")].copy()
        paired = historical.merge(social, on=key, suffixes=("_h", "_s"), validate="one_to_one")
        if len(paired) != len(historical) or len(paired) != len(social):
            raise ValueError("Amostras históricas e sociais não são pareadas.")
        paired["delta"] = (paired.alvo_log_100_s-paired.previsao_log_100_s).abs() - (paired.alvo_log_100_h-paired.previsao_log_100_h).abs()
        agg = paired.groupby("municipio_codigo").delta.agg(["sum", "count"]).to_numpy()
        rng = np.random.default_rng(seed)
        estimates = []
        for _ in range(iterations):
            draw = agg[rng.integers(0, len(agg), len(agg))]
            estimates.append(draw[:,0].sum()/draw[:,1].sum())
        rows.append({"coorte": cohort, "n": len(paired), "municipios": len(agg),
            "delta_mae_social_menos_historico": paired.delta.mean(),
            "ic95_inferior": np.quantile(estimates,.025), "ic95_superior": np.quantile(estimates,.975),
            "reamostragens": iterations})
    return pd.DataFrame(rows)


def error_groups(predictions, group_columns):
    data = predictions.copy()
    data["erro_absoluto"] = (data.alvo_log_100-data.previsao_log_100).abs()
    data["erro_quadrado"] = (data.alvo_log_100-data.previsao_log_100)**2
    result = data.groupby(group_columns).agg(n=("erro_absoluto","size"),
        mae_log_100=("erro_absoluto","mean"), mse=("erro_quadrado","mean")).reset_index()
    result["rmse_log_100"] = np.sqrt(result.pop("mse"))
    return result
