import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from app.data import (TEMPORAL_DOMAIN, apply_prediction_filters, color,
    build_prediction_temporal_chart, build_temporal_chart, descriptive_layer,
    map_features, map_points, map_positions, prediction_layer,
    prediction_metric_layer, scale_bounds, simplified_geometry,
    value_definition)
from src.indicators import SERIES, attach_social
from src.population import attach_population_features
from src.validation import cluster_bootstrap


class TemporalValidationTests(unittest.TestCase):
    def test_same_publication_year_uses_latest_census_and_never_anticipates(self):
        with TemporaryDirectory() as tmp:
            for field, code in SERIES.items():
                pd.DataFrame({'SERCODIGO':[code]*3,'NIVNOME':['Municípios']*3,
                    'TERCODIGO':['1100015']*3,'VALDATA':['2010-01-01','1991-01-01','2000-01-01'],
                    'VALVALOR':[.8,.2,.4]}).to_csv(Path(tmp)/f'{code}.csv',index=False)
            panel=pd.DataFrame({'municipio_codigo':['1100015']*2,'ano':[2012,2013]})
            output=attach_social(panel,tmp,publication_years={1991:2013,2000:2013,2010:2013})
            self.assertTrue(pd.isna(output.gini.iloc[0]))
            self.assertEqual(output.gini.iloc[1],.8)
            self.assertEqual(output.gini_ano_referencia.iloc[1],2010)

    def test_population_2022_not_available_before_release_2023(self):
        panel=pd.DataFrame({'municipio_codigo':['1100015']*2,'ano':[2022,2023]})
        population=pd.DataFrame({'municipio_codigo':['1100015']*2,'ano':[2021,2022],
            'populacao':[50000,48000],'fonte_tabela':['6579','4714'],'populacao_disponivel_desde':[2021,2023]})
        data=attach_population_features(panel,population)
        self.assertEqual(data.populacao.tolist(),[50000,48000])
        self.assertEqual(data.populacao_ano_referencia.tolist(),[2021,2022])

    def test_cluster_bootstrap_pairs_and_detects_known_gain(self):
        rows=[]
        for city in ['1100015','1100023']:
            for year in [2020,2021]:
                for exp,pred in [('historico',4.),('social',2.)]:
                    rows.append({'municipio_codigo':city,'ano':year,'janela':'2020-2021',
                        'coorte':'todos','modelo':'random_forest','divisao':'test','experimento':exp,
                        'alvo_log_100':0.,'previsao_log_100':pred})
        frame=pd.DataFrame(rows)
        result=cluster_bootstrap(frame,iterations=50).iloc[0]
        self.assertEqual(result.delta_mae_social_menos_historico,-2)
        self.assertEqual(result.ic95_superior,-2)
        with self.assertRaises(ValueError): cluster_bootstrap(frame.iloc[:-1],iterations=10)


class AppDataTests(unittest.TestCase):
    def test_yellow_red_palette_increases_intensity_with_value(self):
        low = color(0, 0, 100, sequential_palette='yellow_red')
        middle = color(50, 0, 100, sequential_palette='yellow_red')
        high = color(100, 0, 100, sequential_palette='yellow_red')
        self.assertEqual(low, [255, 247, 188, 220])
        self.assertEqual(middle, [254, 153, 41, 220])
        self.assertEqual(high, [177, 0, 38, 220])

    def test_value_definitions_explain_observed_and_predicted_metrics(self):
        label, explanation = value_definition('observed', 'taxa_homicidios')
        self.assertIn('100 mil', label)
        self.assertIn('não é o total absoluto', explanation)

        label, explanation = value_definition('prediction', 'risco')
        self.assertIn('probabilidade', label.lower())
        self.assertIn('não é certeza', explanation)

        with self.assertRaises(ValueError):
            value_definition('prediction', 'desconhecida')

    def test_reference_filter_and_absence_are_not_zero(self):
        frame=pd.DataFrame({'municipio_codigo':['1100015','1100023'],'ano':[2024,2024],
            'gini':[.5,np.nan],'gini_ano_referencia':[2010,np.nan]})
        exact=descriptive_layer(frame,2024,'gini',True)
        retained=descriptive_layer(frame,2024,'gini')
        self.assertTrue(exact.valor.isna().all())
        self.assertEqual(retained.valor.iloc[0],.5)
        self.assertTrue(pd.isna(retained.valor.iloc[1]))

    def test_geometry_remains_without_data_and_input_not_mutated(self):
        geo={'features':[{'properties':{'codarea':'1100015'},'geometry':{'type':'Polygon','coordinates':[]}},
            {'properties':{'codarea':'1100023'},'geometry':{'type':'Polygon','coordinates':[]}}]}
        frame=pd.DataFrame({'municipio_codigo':['1100015'],'valor':[0.],'referencia':[2024],'municipio':['A']})
        result=map_features(geo,frame,0,100)
        self.assertEqual(len(result['features']),2)
        self.assertNotEqual(result['features'][0]['properties']['cor'],color(np.nan,0,100))
        self.assertEqual(result['features'][1]['properties']['valor_texto'],'Sem dados')
        self.assertNotIn('cor',geo['features'][0]['properties'])

    def test_map_can_omit_missing_values_for_webgl_payload(self):
        geo={'features':[{'properties':{'codarea':'1100015'},'geometry':{'type':'Polygon','coordinates':[]}},
            {'properties':{'codarea':'1100023'},'geometry':{'type':'Polygon','coordinates':[]}}]}
        frame=pd.DataFrame({'municipio_codigo':['1100015'],'valor':[0.],'referencia':[2024],
            'municipio':['A']})
        result=map_features(geo,frame,0,100,include_missing=False)
        self.assertEqual([f['properties']['codigo'] for f in result['features']],['1100015'])

    def test_simplified_geometry_preserves_feature_identity(self):
        geo={'type':'FeatureCollection','features':[{
            'type':'Feature', 'properties':{'codarea':'1100015'}, 'geometry':{
                'type':'Polygon', 'coordinates':[[[0,0],[1,0],[1,0.01],[1,1],[0,1],[0,0]]]
            }
        }]}
        result=simplified_geometry(geo,tolerance=0.1)
        self.assertEqual(result['features'][0]['properties']['codarea'],'1100015')
        self.assertEqual(result['features'][0]['geometry']['type'],'Polygon')
        ring=result['features'][0]['geometry']['coordinates'][0]
        self.assertEqual(ring[0],ring[-1])
        self.assertGreaterEqual(len(ring),4)

    def test_map_points_has_position_and_preserves_value(self):
        geo={'features':[{'properties':{'codarea':'1100015'},'geometry':{
            'type':'Polygon','coordinates':[[[0,0],[1,0],[1,1],[0,1],[0,0]]]
        }}]}
        frame=pd.DataFrame({'municipio_codigo':['1100015'],'valor':[12.5],
            'referencia':[2024],'municipio':['A'],'situacao':['Observação do ano']})
        result=map_points(geo,frame,0,100)
        self.assertEqual(len(result),1)
        self.assertEqual(result[0]['position'],[.4,.4])
        self.assertEqual(result[0]['valor_texto'],'12.50')

    def test_map_positions_preserve_brazilian_extent(self):
        geo={'features':[
            {'properties':{'codarea':'1100015'},'geometry':{
                'type':'Polygon','coordinates':[[[-70,-30],[-69,-30],[-69,-29],[-70,-29],[-70,-30]]]
            }},
            {'properties':{'codarea':'5300108'},'geometry':{
                'type':'Polygon','coordinates':[[[-48,-16],[-47,-16],[-47,-15],[-48,-15],[-48,-16]]]
            }},
            {'properties':{'codarea':'1500107'},'geometry':{
                'type':'Polygon','coordinates':[[[-35,0],[-34,0],[-34,1],[-35,1],[-35,0]]]
            }},
        ]}
        frame=pd.DataFrame({'municipio_codigo':['1100015','5300108','1500107']})
        positions=map_positions(geo,frame)
        longitudes=[row['position'][0] for row in positions]
        latitudes=[row['position'][1] for row in positions]
        self.assertGreater(max(longitudes)-min(longitudes),30)
        self.assertGreater(max(latitudes)-min(latitudes),20)

    def test_robust_scale_is_symmetric_for_variation(self):
        frame=pd.DataFrame({'valor':list(np.linspace(-10,10,100))+[1000.]})
        low,high=scale_bounds(frame,-100,100,diverging=True,robust=True)
        self.assertEqual(low,-high)
        self.assertLess(high,100)

    def test_prediction_filters_combine_risk_rank_and_population(self):
        frame=pd.DataFrame({
            'faixa_risco':['baixo','alto','muito_alto'],
            'percentil_ranking_taxa':[.2,.8,.95],
            'populacao':[40_000,60_000,200_000],
        })
        result=apply_prediction_filters(frame,'alto',.75,50_000)
        self.assertEqual(len(result),1)
        self.assertEqual(result.percentil_ranking_taxa.iloc[0],.8)

    def test_temporal_chart_has_fixed_2000_2030_domain(self):
        frame=pd.DataFrame({'ano':[2000,2010,2024],'taxa_homicidios':[10.,20.,30.]})
        chart=build_temporal_chart(frame,'taxa_homicidios','Taxa de homicídios')
        spec=chart.to_dict()
        self.assertEqual(TEMPORAL_DOMAIN,(2000,2030))
        self.assertEqual(spec['encoding']['x']['scale']['domain'],[2000,2030])
        self.assertEqual(spec['encoding']['x']['field'],'ano')
        self.assertEqual(spec['encoding']['y']['field'],'taxa_homicidios')

    def test_prediction_temporal_chart_layers_observed_forecast_and_interval(self):
        history = pd.DataFrame({
            'ano': [2020, 2021, 2022, 2023, 2024],
            'taxa_homicidios': [20., 22., 19., 21., 18.],
        })
        prediction = pd.Series({
            'ano': 2024,
            'ano_futuro_inicio': 2025,
            'ano_futuro': 2026,
            'taxa_prevista': 17.,
            'intervalo_inferior_taxa_media_2_anos': 8.,
            'intervalo_superior_taxa_media_2_anos': 28.,
        })
        chart = build_prediction_temporal_chart(history, prediction)
        spec = chart.to_dict()
        self.assertEqual(len(spec['layer']), 5)
        self.assertEqual(spec['layer'][0]['encoding']['y']['field'], 'taxa_homicidios')
        self.assertEqual(spec['layer'][1]['encoding']['y']['field'], 'taxa_prevista')
        self.assertEqual(
            spec['layer'][2]['encoding']['y']['field'], 'intervalo_inferior'
        )
        self.assertEqual(spec['layer'][0]['encoding']['x']['scale']['domain'], [2000, 2030])

    def test_prediction_layer_rejects_training_values(self):
        indicators=pd.DataFrame({'municipio_codigo':['1100015'],'municipio':['A'],'uf':['RO'],'ano':[2020]})
        predictions=pd.DataFrame({'municipio_codigo':['1100015'],'ano':[2020],'coorte':['todos'],'divisao':['train']})
        with self.assertRaises(ValueError): prediction_layer(indicators,predictions,2020,'todos')

    def test_prediction_layer_supports_direct_future_rate_target(self):
        indicators = pd.DataFrame(
            {
                'municipio_codigo': ['1100015'],
                'municipio': ['A'],
                'uf': ['RO'],
                'ano': [2020],
            }
        )
        predictions = pd.DataFrame(
            {
                'municipio_codigo': ['1100015'],
                'ano': [2020],
                'coorte': ['todos'],
                'divisao': ['test'],
                'previsao_percentual': [10.0],
                'taxa_atual': [20.0],
                'taxa_prevista': [22.0],
                'alvo_percentual': [15.0],
                'modelo': ['random_forest'],
                'experimento': ['social'],
                'ano_futuro': [2021],
                'janela': ['2021-2021'],
                'alvo_log_taxa_100': [100 * np.log1p(25.0)],
            }
        )
        result = prediction_layer(indicators, predictions, 2020, 'todos')
        self.assertAlmostEqual(result.taxa_observada_alvo.iloc[0], 25.0)

    def test_prediction_layer_supports_final_smoothed_target_and_risk(self):
        predictions = pd.DataFrame(
            {
                'municipio_codigo': ['1100015'], 'municipio': ['A'],
                'uf_codigo': ['11'], 'ano': [2020], 'ano_futuro': [2023],
                'coorte': ['todos'], 'divisao': ['test'],
                'taxa_media_3_anos_atual': [20.0],
                'taxa_media_3_anos_prevista': [22.0],
                'alvo_taxa_media_3_anos': [25.0],
                'previsao_variacao_media_3_anos_percentual': [10.0],
                'alvo_variacao_media_3_anos_percentual': [25.0],
                'probabilidade_alto_risco': [0.8],
            }
        )
        result = prediction_layer(pd.DataFrame(), predictions, 2020, 'todos')
        self.assertAlmostEqual(result.taxa_observada_alvo.iloc[0], 25.0)
        self.assertAlmostEqual(prediction_metric_layer(result, 'risco').valor.iloc[0], 0.8)
        self.assertAlmostEqual(prediction_metric_layer(result, 'taxa').valor.iloc[0], 22.0)

    def test_prediction_layer_requires_window_when_origin_has_multiple_windows(self):
        predictions = pd.DataFrame(
            {
                'municipio_codigo': ['1100015', '1100015'], 'municipio': ['A', 'A'],
                'uf_codigo': ['11', '11'], 'ano': [2020, 2020],
                'ano_futuro': [2022, 2023], 'janela': ['2019-2020', '2020-2021'],
                'coorte': ['todos', 'todos'], 'divisao': ['test', 'test'],
                'taxa_media_3_anos_atual': [20.0, 20.0],
                'taxa_media_3_anos_prevista': [22.0, 23.0],
                'alvo_taxa_media_3_anos': [25.0, 26.0],
                'previsao_variacao_media_3_anos_percentual': [10.0, 15.0],
                'alvo_variacao_media_3_anos_percentual': [25.0, 30.0],
                'probabilidade_alto_risco': [0.8, 0.9],
            }
        )
        with self.assertRaises(ValueError):
            prediction_layer(pd.DataFrame(), predictions, 2020, 'todos')
        result = prediction_layer(pd.DataFrame(), predictions, 2020, 'todos', '2020-2021')
        self.assertEqual(result.janela.iloc[0], '2020-2021')
