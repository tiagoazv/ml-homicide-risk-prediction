import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.models import evaluate_by_population_size, run_experiment
from src.prepare_data import build_model_dataset
from tests.test_prepare_data import make_rates


class ModelTests(unittest.TestCase):
    def test_experiment_returns_metrics_and_predictions(self):
        rates = make_rates(
            {
                "0000001": range(2000, 2025),
                "0000002": range(2000, 2025),
                "0000003": range(2000, 2025),
            }
        )
        model_data = build_model_dataset(rates)

        with TemporaryDirectory() as model_dir:
            metrics, predictions = run_experiment(
                model_data, model_output_dir=model_dir
            )
            self.assertEqual(len(list(Path(model_dir).glob("*.joblib"))), 4)

        self.assertIsInstance(metrics, pd.DataFrame)
        self.assertIsInstance(predictions, pd.DataFrame)
        self.assertEqual(
            set(metrics["modelo"]),
            {
                "baseline_zero",
                "baseline_last_value",
                "ridge",
                "random_forest",
            },
        )
        self.assertEqual(set(metrics["divisao"]), {"train", "validation", "test"})
        self.assertEqual(len(predictions), len(model_data) * 4)
        self.assertTrue(predictions["previsao_percentual_estabilizada"].notna().all())
        self.assertFalse(evaluate_by_population_size(predictions).empty)


if __name__ == "__main__":
    unittest.main()
