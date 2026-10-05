import unittest

import numpy as np
import pandas as pd

from src.population import POPULATION_SIZE_LABELS, filter_by_population
from src.prepare_data import (
    TEMPORAL_FEATURE_COLUMNS,
    TARGET_LOG_COLUMN,
    TARGET_PERCENT_COLUMN,
    TARGET_RATE_LOG_COLUMN,
    OFFICIAL_COUNT_TARGET_COLUMN,
    build_model_dataset,
    log_change_to_percent,
    split_by_year,
)


def make_rates(years_by_municipality):
    rows = []
    for code, years in years_by_municipality.items():
        for year in years:
            rows.append(
                {
                    "municipio_codigo": code,
                    "municipio": f"Município {code}",
                    "ano": year,
                    "taxa_homicidios": float(year - 1999),
                }
            )
    return pd.DataFrame(rows)


class PrepareDataTests(unittest.TestCase):
    def test_temporal_features_and_count_proxy_use_origin_population(self):
        rates = make_rates({"0000001": range(2000, 2006)})
        population = pd.DataFrame(
            [
                {
                    "municipio_codigo": "0000001",
                    "ano": year,
                    "populacao": 100_000,
                    "fonte_tabela": "teste",
                }
                for year in range(2000, 2006)
            ]
        )

        model_data = build_model_dataset(rates, population=population)
        row = model_data.loc[model_data["ano"].eq(2002)].iloc[0]

        self.assertTrue(set(TEMPORAL_FEATURE_COLUMNS).issubset(model_data.columns))
        self.assertAlmostEqual(row["taxa_mediana_3_anos"], 2.0)
        self.assertAlmostEqual(row["tendencia_taxa_3_anos"], 1.0)
        self.assertEqual(row["aumentos_3_anos"], 2)
        self.assertAlmostEqual(
            row["homicidios_proxy_futuro"], row["taxa_futuro"]
        )

    def test_target_uses_only_consecutive_future_year(self):
        rates = make_rates(
            {
                "0000001": [2000, 2001, 2002, 2003, 2004],
                "0000002": [2000, 2001, 2003, 2004, 2005],
            }
        )

        model_data = build_model_dataset(rates)

        self.assertEqual(set(model_data["municipio_codigo"]), {"0000001"})
        row = model_data.loc[
            (model_data["municipio_codigo"] == "0000001")
            & (model_data["ano"] == 2002)
        ].iloc[0]
        expected_log = 100 * (np.log1p(4.0) - np.log1p(3.0))
        self.assertAlmostEqual(row[TARGET_LOG_COLUMN], expected_log)
        self.assertAlmostEqual(row[TARGET_PERCENT_COLUMN], 100 / 3)
        self.assertAlmostEqual(row[TARGET_RATE_LOG_COLUMN], 100 * np.log1p(4.0))

    def test_split_has_disjoint_temporal_ranges(self):
        rates = make_rates(
            {
                "0000001": range(2000, 2025),
                "0000002": range(2000, 2025),
            }
        )
        model_data = build_model_dataset(rates)
        splits = split_by_year(model_data)

        self.assertLess(splits["train"]["ano"].max(), splits["validation"]["ano"].min())
        self.assertLess(splits["validation"]["ano"].max(), splits["test"]["ano"].min())
        self.assertEqual(len(model_data), sum(len(split) for split in splits.values()))

    def test_log_change_conversion_round_trip(self):
        values = np.array([-50.0, 0.0, 25.0, 100.0])
        percentages = log_change_to_percent(values)
        self.assertTrue(
            np.allclose(
                percentages,
                np.array([-39.34693403, 0.0, 28.40254167, 171.82818285]),
            )
        )

    def test_population_feature_uses_previous_value_for_missing_year(self):
        rates = make_rates({"0000001": range(2000, 2023)})
        population = pd.DataFrame(
            [
                {
                    "municipio_codigo": "0000001",
                    "ano": year,
                    "populacao": 20_000 + (year - 2000) * 100,
                    "fonte_tabela": "6579",
                }
                for year in range(2001, 2022)
                if year != 2007
            ]
        )

        model_data = build_model_dataset(rates, population=population)
        row = model_data.loc[model_data["ano"].eq(2007)].iloc[0]

        self.assertEqual(row["populacao"], 20_600)
        self.assertFalse(row["populacao_observada"])
        self.assertIn(row["porte_populacional"], POPULATION_SIZE_LABELS)
        self.assertTrue(model_data["log_populacao"].notna().all())

    def test_population_filter_uses_origin_population(self):
        data = pd.DataFrame(
            {
                "municipio_codigo": ["0000001", "0000002"],
                "ano": [2019, 2019],
                "populacao": [49_999, 50_000],
            }
        )

        selected = filter_by_population(data, minimum=50_000)

        self.assertEqual(selected["municipio_codigo"].tolist(), ["0000002"])

    def test_official_sim_counts_are_joined_by_origin_and_future_year(self):
        rates = make_rates({"0000001": range(2000, 2006)})
        population = pd.DataFrame(
            [
                {
                    "municipio_codigo": "0000001",
                    "ano": year,
                    "populacao": 100_000,
                    "fonte_tabela": "teste",
                }
                for year in range(2000, 2006)
            ]
        )
        sim_counts = pd.DataFrame(
            {
                "municipio_codigo": ["0000001"] * 6,
                "ano": range(2000, 2006),
                "homicidios_sim": range(10, 16),
            }
        )

        model_data = build_model_dataset(rates, population, sim_counts)
        row = model_data.loc[model_data["ano"].eq(2002)].iloc[0]

        self.assertEqual(row["homicidios_sim_atual"], 12)
        self.assertEqual(row[OFFICIAL_COUNT_TARGET_COLUMN], 13)
        self.assertAlmostEqual(row["taxa_homicidios_sim_atual"], 12.0)
        self.assertAlmostEqual(row["taxa_homicidios_sim_futuro"], 13.0)


if __name__ == "__main__":
    unittest.main()
