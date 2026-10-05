import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd
from src.indicators import SERIES, attach_social, load_atlas_rates, load_social, municipality_lookup
from src.models_final import evaluate_smoothed_rate_predictions
from src.population import attach_population_features


class IndicatorTests(unittest.TestCase):
    def test_prefix_resolved_not_zero_padded(self):
        with TemporaryDirectory() as tmp:
            registry = Path(tmp) / "names.json"
            registry.write_text(json.dumps([{"id":1100015,"nome":"Alta Floresta D'Oeste"}]))
            lookup = municipality_lookup(pd.DataFrame({"municipio_codigo":["1100015"]}), registry)
            path = Path(tmp)/"rates.csv"
            pd.DataFrame({"Período":["2020-01-15"],"Região ID":["110001"],"Valor":[12]}).to_csv(path,index=False)
            data = load_atlas_rates(path,lookup)
            self.assertEqual(data.municipio_codigo.iloc[0],"1100015")
            self.assertIn("Alta",data.municipio.iloc[0])
            with self.assertRaises(ValueError): load_atlas_rates(path,lookup.iloc[:0])

    def test_social_excludes_aggregates_future_and_other_city(self):
        with TemporaryDirectory() as tmp:
            for column,series in SERIES.items():
                pd.DataFrame({"SERCODIGO":[series]*3,"VALDATA":["2000-01-01","2010-01-01","2000-01-01"],
                    "VALVALOR":[.4,.8,.99],"NIVNOME":["Municípios","Municípios","Brasil"],
                    "TERCODIGO":["1100015","1100015","0"]}).to_csv(Path(tmp)/f"{series}.csv",index=False)
            panel=pd.DataFrame({"municipio_codigo":["1100015"]*4+["1100023"],"ano":[2002,2003,2012,2013,2013]})
            result=attach_social(panel,tmp,release_lag=3)
            self.assertTrue(pd.isna(result.gini.iloc[0]))
            self.assertEqual(result.gini.iloc[1:4].tolist(),[.4,.4,.8])
            self.assertTrue(pd.isna(result.gini.iloc[4]))

    def test_duplicate_and_invalid_social_fail(self):
        with TemporaryDirectory() as tmp:
            p=Path(tmp)/"gini.csv"
            row={"SERCODIGO":"ADH_GINI","VALDATA":"2010-01-01","VALVALOR":.5,"NIVNOME":"Municípios","TERCODIGO":"1100015"}
            pd.DataFrame([row,row]).to_csv(p,index=False)
            with self.assertRaises(ValueError): load_social(p,"ADH_GINI","gini")
            pd.DataFrame([{**row,"VALVALOR":2}]).to_csv(p,index=False)
            with self.assertRaises(ValueError): load_social(p,"ADH_GINI","gini")

    def test_percent_error_uses_inverse_rate(self):
        y=pd.Series([100*np.log(2),100*np.log(5)])
        rates=pd.Series([1.,4.])
        metrics=evaluate_smoothed_rate_predictions(
            y, y.to_numpy(), rates, rates
        )
        self.assertAlmostEqual(metrics['mae_percentual'],0.)

    def test_population_before_first_panel_row(self):
        panel=pd.DataFrame({"municipio_codigo":["1100015"],"ano":[2003]})
        pop=pd.DataFrame({"municipio_codigo":["1100015"],"ano":[2001],"populacao":[50000],"fonte_tabela":["6579"]})
        result=attach_population_features(panel,pop)
        self.assertEqual(result.populacao.iloc[0],50000)
        self.assertFalse(result.populacao_observada.iloc[0])
