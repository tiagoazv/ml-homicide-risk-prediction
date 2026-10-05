import unittest

import numpy as np
import pandas as pd

from src.models_final import evaluate_smoothed_rate_predictions
from src.prepare_data import (
    FEATURE_COLUMNS_BASE,
    SMOOTHED_RATE_LOG_TARGET_COLUMN,
    SMOOTHED_RATE_PERCENT_TARGET_COLUMN,
    SMOOTHED_RATE_TARGET_COLUMN,
    build_smoothed_model_dataset,
    split_by_year,
)
from tests.test_prepare_data import make_rates


class FinalIntegrityTests(unittest.TestCase):
    def _panel(self):
        rates = make_rates({"0000001": range(2000, 2025)})
        population = pd.DataFrame(
            [
                {
                    "municipio_codigo": "0000001",
                    "ano": year,
                    "populacao": 100_000,
                    "fonte_tabela": "teste",
                }
                for year in range(2000, 2025)
            ]
        )
        return build_smoothed_model_dataset(rates, population)

    def test_target_and_derived_percent_follow_their_formulas(self):
        row = self._panel().loc[lambda data: data.ano.eq(2002)].iloc[0]

        current_mean = 2.0
        future_mean = (4.0 + 5.0 + 6.0) / 3
        self.assertAlmostEqual(row[SMOOTHED_RATE_TARGET_COLUMN], future_mean)
        self.assertAlmostEqual(
            row[SMOOTHED_RATE_LOG_TARGET_COLUMN], 100 * np.log1p(future_mean)
        )
        self.assertAlmostEqual(
            row[SMOOTHED_RATE_PERCENT_TARGET_COLUMN],
            100 * (future_mean - current_mean) / current_mean,
        )

    def test_missing_future_year_is_not_treated_as_zero(self):
        rates = make_rates({"0000001": [2000, 2001, 2002, 2003, 2005, 2006]})
        panel = build_smoothed_model_dataset(rates)

        self.assertNotIn(2002, set(panel.ano))

    def test_features_do_not_include_future_or_target_columns(self):
        panel = self._panel()
        forbidden = {
            SMOOTHED_RATE_TARGET_COLUMN,
            SMOOTHED_RATE_LOG_TARGET_COLUMN,
            SMOOTHED_RATE_PERCENT_TARGET_COLUMN,
            "taxa_futura_1",
            "taxa_futura_2",
            "taxa_futura_3",
        }
        self.assertTrue(set(FEATURE_COLUMNS_BASE).issubset(panel.columns))
        self.assertTrue(set(FEATURE_COLUMNS_BASE).isdisjoint(forbidden))

    def test_rate_metrics_are_calculated_on_the_rate_scale(self):
        actual_rates = pd.Series([5.0, 7.0])
        predicted_rates = np.array([5.0, 8.0])
        actual_log = pd.Series(100 * np.log1p(actual_rates))
        predicted_log = 100 * np.log1p(predicted_rates)
        metrics = evaluate_smoothed_rate_predictions(
            actual_log,
            predicted_log,
            actual_rates,
            pd.Series([3.0, 7.0]),
        )

        self.assertAlmostEqual(metrics["mae_taxa"], 0.5)
        self.assertAlmostEqual(metrics["mae_percentual"], 7.1428571429)

    def test_temporal_splits_are_disjoint(self):
        splits = split_by_year(self._panel(), train_end=2014, validation_end=2016)

        self.assertLess(splits["train"].ano.max(), splits["validation"].ano.min())
        self.assertLess(splits["validation"].ano.max(), splits["test"].ano.min())


if __name__ == "__main__":
    unittest.main()
