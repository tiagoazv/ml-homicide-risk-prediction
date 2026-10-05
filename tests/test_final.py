import unittest

import pandas as pd

from app.data import future_prediction_layer, prediction_metric_layer
from src.count_sensibilidade import (
    COUNT_FEATURES,
    EXPOSURE_COLUMN,
    ExposurePoissonRegressor,
)
from src.enrich_previsoes import _add_rank_and_risk_band, _conformal_quantile
from src.forecast_futura import build_future_feature_panel
from src.models_final import build_smoothed_rate_models
from src.prepare_data import build_smoothed_model_dataset
from tests.test_prepare_data import make_rates


def make_final_panel():
    rates = make_rates({'0000001': range(2000, 2025)})
    population = pd.DataFrame([
        {
            'municipio_codigo': '0000001',
            'ano': year,
            'populacao': 100_000,
            'fonte_tabela': 'teste',
        }
        for year in range(2000, 2025)
    ])
    return build_smoothed_model_dataset(
        rates, population, horizon=2, history_years=5
    )


class V11Tests(unittest.TestCase):
    def test_tuned_rate_models_preserve_model_families(self):
        models = build_smoothed_rate_models(
            horizon=2,
            model_params={
                "ridge": {"alpha": 10.0},
                "random_forest": {
                    "n_estimators": 20,
                    "min_samples_leaf": 5,
                    "max_features": "sqrt",
                },
                "hist_gradient_boosting": {
                    "max_iter": 50,
                    "learning_rate": 0.1,
                    "max_leaf_nodes": 7,
                },
            },
        )
        self.assertEqual(
            set(models),
            {
                "baseline_media_2_anos",
                "ridge",
                "random_forest",
                "hist_gradient_boosting",
            },
        )
        self.assertEqual(models["ridge"].named_steps["model"].alpha, 10.0)

    def test_poisson_uses_two_year_population_exposure(self):
        panel = make_final_panel().copy()
        panel[EXPOSURE_COLUMN] = panel.populacao * 2 / 100_000
        panel["target_count"] = panel.taxa_futura_media_2_anos * panel[EXPOSURE_COLUMN]
        model = ExposurePoissonRegressor(
            alpha=0.1,
            feature_columns=COUNT_FEATURES,
        )
        model.fit(panel[COUNT_FEATURES + [EXPOSURE_COLUMN]], panel.target_count)
        predictions = model.predict(panel[COUNT_FEATURES + [EXPOSURE_COLUMN]])
        self.assertTrue((predictions >= 0).all())

    def test_rank_risk_band_and_conformal_quantile(self):
        data = pd.DataFrame(
            {
                "coorte": ["todos", "todos", "todos"],
                "janela": ["2015-2016"] * 3,
                "ano": [2015] * 3,
                "taxa_prevista": [30.0, 20.0, 10.0],
                "probabilidade_alto_risco_calibrada": [0.9, 0.6, 0.1],
            }
        )
        result = _add_rank_and_risk_band(data, "taxa_prevista")
        self.assertEqual(result.ranking_taxa_futura.tolist(), [1, 2, 3])
        self.assertEqual(result.faixa_risco.tolist(), ["muito_alto", "alto", "baixo"])
        self.assertEqual(_conformal_quantile([1.0, 2.0, 3.0], 0.9), 3.0)

    def test_app_exposes_final_map_representations(self):
        frame = pd.DataFrame(
            {
                "taxa_prevista": [20.0],
                "probabilidade_alto_risco_calibrada": [0.8],
                "percentil_ranking_taxa": [0.9],
                "intervalo_inferior_taxa_media_2_anos": [10.0],
                "intervalo_superior_taxa_media_2_anos": [30.0],
            }
        )
        self.assertEqual(prediction_metric_layer(frame, "taxa").valor.iloc[0], 20.0)
        self.assertEqual(prediction_metric_layer(frame, "faixa").valor.iloc[0], 0.8)
        self.assertEqual(prediction_metric_layer(frame, "ranking").valor.iloc[0], 0.9)
        self.assertEqual(prediction_metric_layer(frame, "incerteza").valor.iloc[0], 20.0)

    def test_future_feature_panel_requires_five_consecutive_years(self):
        rates = pd.DataFrame(
            {
                "municipio_codigo": ["1100015"] * 5,
                "municipio": ["A"] * 5,
                "uf_codigo": ["11"] * 5,
                "ano": [2020, 2021, 2022, 2023, 2024],
                "taxa_homicidios": [10.0, 12.0, 11.0, 13.0, 14.0],
            }
        )
        population = pd.DataFrame(
            {
                "municipio_codigo": ["1100015"] * 5,
                "ano": [2020, 2021, 2022, 2023, 2024],
                "populacao": [50_000] * 5,
                "fonte_tabela": ["teste"] * 5,
            }
        )
        result = build_future_feature_panel(rates, population)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.ano_futuro_inicio.iloc[0], 2025)
        self.assertEqual(result.ano_futuro_final.iloc[0], 2026)
        self.assertEqual(result.taxa_media_5_anos.iloc[0], 12.0)
        self.assertNotIn("taxa_futura_media_2_anos", result)

    def test_future_prediction_layer_rejects_retrospective_rows(self):
        frame = pd.DataFrame(
            {
                "municipio_codigo": ["1100015"],
                "ano": [2024],
                "coorte": ["todos"],
                "divisao": ["test"],
            }
        )
        with self.assertRaises(ValueError):
            future_prediction_layer(frame, 2024, "todos")


if __name__ == "__main__":
    unittest.main()
