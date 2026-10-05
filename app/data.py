"""Leitura local e junções explícitas para a aplicação, sem treinar modelos."""
import json
import re
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
UF = dict(zip(['11','12','13','14','15','16','17','21','22','23','24','25','26','27','28','29','31','32','33','35','41','42','43','50','51','52','53'],
              ['RO','AC','AM','RR','PA','AP','TO','MA','PI','CE','RN','PB','PE','AL','SE','BA','MG','ES','RJ','SP','PR','SC','RS','MS','MT','GO','DF']))
LAYERS = {
    'Taxa de homicídios': ('taxa_homicidios', 'por 100 mil habitantes', 0, 100),
    'População': ('populacao', 'habitantes', 0, 12_000_000),
    'GINI': ('gini', 'índice de 0 a 1', 0, 1),
    'IDHM': ('idhm', 'índice de 0 a 1', 0, 1),
    'Extrema pobreza (PIND)': ('pind', '%', 0, 100),
    'Analfabetismo — 15 anos ou mais': ('analfabetismo_15_mais', '%', 0, 100),
}
MAP_GEOMETRY_TOLERANCE = 0.05
TEMPORAL_DOMAIN = (2000, 2030)
VALUE_DEFINITIONS = {
    'observed': {
        'taxa_homicidios': (
            'Taxa de homicídios (por 100 mil habitantes)',
            'Taxa relativa de homicídios; não é o total absoluto de casos.',
        ),
        'populacao': (
            'População (habitantes)',
            'Estimativa da população municipal no ano de referência.',
        ),
        'gini': (
            'GINI (índice de 0 a 1)',
            'Mede a desigualdade de renda; valores maiores indicam maior desigualdade.',
        ),
        'idhm': (
            'IDHM (índice de 0 a 1)',
            'Mede o desenvolvimento humano municipal; valores maiores indicam maior desenvolvimento.',
        ),
        'pind': (
            'Extrema pobreza — PIND (%)',
            'Percentual da população em situação de extrema pobreza.',
        ),
        'analfabetismo_15_mais': (
            'Analfabetismo — 15 anos ou mais (%)',
            'Percentual da população com 15 anos ou mais que não sabe ler e escrever.',
        ),
    },
    'prediction': {
        'taxa': (
            'Taxa média futura prevista (por 100 mil habitantes)',
            'Média estimada da taxa para o horizonte selecionado; não é contagem de casos.',
        ),
        'faixa': (
            'Probabilidade calibrada de alto risco (0 a 1)',
            'Probabilidade de pertencer ao grupo de maior risco; não é certeza de ocorrência.',
        ),
        'risco': (
            'Probabilidade calibrada de alto risco (0 a 1)',
            'Probabilidade de pertencer ao grupo de maior risco; não é certeza de ocorrência.',
        ),
        'ranking': (
            'Percentil do ranking da taxa futura (0 a 1)',
            'Posição relativa na coorte; valores maiores indicam taxas futuras previstas maiores.',
        ),
        'incerteza': (
            'Largura do intervalo de previsão (por 100 mil)',
            'Diferença entre os limites do intervalo; valores maiores indicam maior incerteza.',
        ),
        'variacao': (
            'Variação percentual futura prevista (%)',
            'Aumento ou redução percentual estimada da taxa futura em relação à taxa atual.',
        ),
    },
}


def value_definition(mode, key):
    """Retorna o rótulo e a interpretação da métrica exibida no mapa."""

    try:
        return VALUE_DEFINITIONS[mode][key]
    except KeyError as exc:
        raise ValueError(f'Definição de valor desconhecida: {mode}/{key}') from exc


def _simplify_ring(points, tolerance):
    """Simplifica um anel fechado para visualização em escala nacional."""
    if len(points) <= 4:
        return points
    closed = points[0] == points[-1]
    core = points[:-1] if closed else points
    if len(core) <= 2:
        return points

    def simplify(sequence):
        if len(sequence) <= 2:
            return sequence
        start, end = sequence[0], sequence[-1]
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        denominator = (dx * dx + dy * dy) ** 0.5
        largest = -1.0
        split = 0
        for index, point in enumerate(sequence[1:-1], start=1):
            distance = (
                abs(dy * point[0] - dx * point[1]
                    + end[0] * start[1] - end[1] * start[0])
                / denominator
                if denominator else 0.0
            )
            if distance > largest:
                largest, split = distance, index
        if largest <= tolerance:
            return [start, end]
        left = simplify(sequence[:split + 1])
        right = simplify(sequence[split:])
        return left[:-1] + right

    simplified = simplify(core)
    if closed:
        simplified.append(simplified[0])
    return simplified if len(simplified) >= 4 else points


def _simplify_geometry(geometry, tolerance=MAP_GEOMETRY_TOLERANCE):
    """Retorna uma cópia leve da geometria, mantendo o tipo GeoJSON."""
    geometry_type = geometry['type']
    coordinates = geometry['coordinates']
    if geometry_type == 'Polygon':
        coordinates = [_simplify_ring(ring, tolerance) for ring in coordinates]
    elif geometry_type == 'MultiPolygon':
        coordinates = [
            [_simplify_ring(ring, tolerance) for ring in polygon]
            for polygon in coordinates
        ]
    return {'type': geometry_type, 'coordinates': coordinates}


def simplified_geometry(geometry, tolerance=MAP_GEOMETRY_TOLERANCE):
    """Simplifica a FeatureCollection sem alterar propriedades ou códigos."""
    return {
        'type': geometry.get('type', 'FeatureCollection'),
        'features': [
            {
                **feature,
                'geometry': _simplify_geometry(feature['geometry'], tolerance),
            }
            for feature in geometry['features']
        ],
    }


def load_data(root=ROOT):
    descriptive_folder = root/'data/processed/v5'
    prediction_folder = root/'data/processed/v11'
    descriptive_files = {
        'indicadores_mapa': 'indicadores_mapa.csv',
    }
    prediction_files = {
        'previsoes_mapa': 'previsoes_mapa.csv',
        'metricas_janelas': 'metricas_taxa_suavizada_janelas.csv',
        'selecao_validacao': 'selecao_taxa_suavizada_validacao.csv',
        'metricas_risco': 'metricas_risco_alto_janelas.csv',
        'selecao_risco_validacao': 'selecao_risco_alto_validacao.csv',
    }
    csv_types = {'municipio_codigo':'string', 'uf_codigo':'string'}
    tables = {
        name: pd.read_csv(descriptive_folder/file, dtype=csv_types)
        for name, file in descriptive_files.items()
    }
    tables.update({
        name: pd.read_csv(prediction_folder/file, dtype=csv_types)
        for name, file in prediction_files.items()
    })
    future_path = prediction_folder/'previsoes_futuras_2024_2026.csv'
    tables['previsoes_futuras'] = (
        pd.read_csv(future_path, dtype=csv_types)
        if future_path.exists()
        else pd.DataFrame()
    )
    tables['indicadores_mapa']['uf'] = tables['indicadores_mapa'].uf_codigo.map(UF)
    tables['previsoes_mapa']['uf'] = tables['previsoes_mapa'].uf_codigo.map(UF)
    if not tables['previsoes_futuras'].empty:
        tables['previsoes_futuras']['uf'] = tables['previsoes_futuras'].uf_codigo.map(UF)
    tables['geometry'] = json.loads((root/'data/geo/municipios_2022.geojson').read_text())
    tables['geometry_map'] = simplified_geometry(tables['geometry'])
    tables['geometry_audit'] = json.loads((descriptive_folder/'auditoria_geometria.json').read_text())
    tables['config'] = json.loads((prediction_folder/'config.json').read_text())
    return tables


def descriptive_layer(indicators, year, field, exact=False):
    result = indicators.loc[indicators.ano.eq(year)].copy()
    ref=f'{field}_ano_referencia'
    result['referencia'] = result[ref] if ref in result else result.ano.where(result[field].notna())
    result['valor'] = result[field]
    if exact:
        result.loc[result.referencia.ne(year),'valor']=np.nan
    result['situacao'] = np.where(result.valor.isna(),'Sem observação para este filtro',
        np.where(result.referencia.eq(year),'Observação do ano','Última referência anterior'))
    return result


def build_temporal_chart(frame, field, title, unit=None):
    """Cria a série temporal com domínio fixo e sem preencher ausências."""

    if field not in frame or 'ano' not in frame:
        raise ValueError('A série precisa conter ano e a variável selecionada.')
    series = frame[['ano', field]].copy()
    series['ano'] = pd.to_numeric(series['ano'], errors='coerce')
    series[field] = pd.to_numeric(series[field], errors='coerce')
    series = series.dropna(subset=['ano', field]).sort_values('ano')
    y_title = title if unit is None else f'{title} ({unit})'
    return (
        alt.Chart(series)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                'ano:Q',
                title='Ano',
                scale=alt.Scale(domain=list(TEMPORAL_DOMAIN), clamp=True),
                axis=alt.Axis(format='d', tickCount=16),
            ),
            y=alt.Y(f'{field}:Q', title=y_title),
            tooltip=[
                alt.Tooltip('ano:Q', title='Ano', format='d'),
                alt.Tooltip(f'{field}:Q', title=title, format='.2f'),
            ],
        )
        .properties(height=300)
    )


def build_prediction_temporal_chart(frame, prediction, title='Taxa de homicídios',
        unit='por 100 mil habitantes'):
    """Combina histórico observado com a média prevista e sua incerteza.

    A previsão representa a média de todo o horizonte, e não uma previsão
    independente para cada ano. Por isso, a linha futura é horizontal entre o
    primeiro e o último ano do horizonte.
    """

    required = {'ano', 'taxa_homicidios'}
    if not required.issubset(frame.columns):
        raise ValueError('O histórico precisa conter ano e taxa_homicidios.')
    if isinstance(prediction, pd.DataFrame):
        if len(prediction) != 1:
            raise ValueError('A previsão temporal deve conter exatamente uma linha.')
        prediction = prediction.iloc[0]
    required_prediction = {
        'ano', 'ano_futuro_inicio', 'ano_futuro', 'taxa_prevista',
    }
    missing = required_prediction.difference(prediction.index)
    if missing:
        raise ValueError(f'Colunas ausentes na previsão temporal: {sorted(missing)}')

    history = frame[['ano', 'taxa_homicidios']].copy()
    history['ano'] = pd.to_numeric(history['ano'], errors='coerce')
    history['taxa_homicidios'] = pd.to_numeric(
        history['taxa_homicidios'], errors='coerce'
    )
    origin = int(prediction['ano'])
    history = history.loc[history.ano.le(origin)].dropna().sort_values('ano')
    if history.empty:
        raise ValueError('Não há taxas observadas anteriores à origem da previsão.')

    start = int(prediction['ano_futuro_inicio'])
    end = int(prediction['ano_futuro'])
    predicted = float(prediction['taxa_prevista'])
    if start > end or not np.isfinite(predicted):
        raise ValueError('O intervalo ou a taxa prevista são inválidos.')
    forecast = pd.DataFrame({
        'ano': [start, end],
        'taxa_prevista': [predicted, predicted],
    })
    layers = []
    x_encoding = alt.X(
        'ano:Q',
        title='Ano',
        scale=alt.Scale(domain=list(TEMPORAL_DOMAIN), clamp=True),
        axis=alt.Axis(format='d', tickCount=16),
    )
    y_title = title if unit is None else f'{title} ({unit})'
    layers.append(
        alt.Chart(history)
        .mark_line(point=True, color='#64748b')
        .encode(
            x=x_encoding,
            y=alt.Y('taxa_homicidios:Q', title=y_title),
            tooltip=[
                alt.Tooltip('ano:Q', title='Ano', format='d'),
                alt.Tooltip('taxa_homicidios:Q', title='Taxa observada', format='.2f'),
            ],
        )
    )
    layers.append(
        alt.Chart(forecast)
        .mark_line(point=True, color='#d94801', strokeDash=[6, 4])
        .encode(
            x=x_encoding,
            y=alt.Y('taxa_prevista:Q', title=y_title),
            tooltip=[
                alt.Tooltip('ano:Q', title='Ano do horizonte', format='d'),
                alt.Tooltip('taxa_prevista:Q', title='Média prevista', format='.2f'),
            ],
        )
    )

    lower = pd.to_numeric(
        prediction.get('intervalo_inferior_taxa_media_2_anos', np.nan),
        errors='coerce',
    )
    upper = pd.to_numeric(
        prediction.get('intervalo_superior_taxa_media_2_anos', np.nan),
        errors='coerce',
    )
    if pd.notna(lower) and pd.notna(upper):
        interval = forecast.assign(
            intervalo_inferior=float(lower), intervalo_superior=float(upper)
        )
        layers.append(
            alt.Chart(interval)
            .mark_area(opacity=0.16, color='#d94801')
            .encode(
                x=x_encoding,
                y=alt.Y('intervalo_inferior:Q', title=y_title),
                y2='intervalo_superior:Q',
                tooltip=[
                    alt.Tooltip('intervalo_inferior:Q', title='Limite inferior', format='.2f'),
                    alt.Tooltip('intervalo_superior:Q', title='Limite superior', format='.2f'),
                ],
            )
        )
        for column, label in (
            ('intervalo_inferior', 'Limite inferior'),
            ('intervalo_superior', 'Limite superior'),
        ):
            layers.append(
                alt.Chart(interval)
                .mark_line(color='#d94801', strokeDash=[2, 2], opacity=0.65)
                .encode(
                    x=x_encoding,
                    y=alt.Y(f'{column}:Q', title=y_title),
                    tooltip=[alt.Tooltip(
                        f'{column}:Q', title=label, format='.2f'
                    )],
                )
            )

    observed_target = pd.to_numeric(
        prediction.get('alvo_taxa_media_futura', np.nan), errors='coerce'
    )
    if pd.notna(observed_target):
        actual = pd.DataFrame({
            'ano': [(start + end) / 2],
            'taxa_observada_horizonte': [float(observed_target)],
        })
        layers.append(
            alt.Chart(actual)
            .mark_point(size=110, shape='diamond', color='#1d4ed8')
            .encode(
                x=x_encoding,
                y=alt.Y('taxa_observada_horizonte:Q', title=y_title),
                tooltip=[alt.Tooltip(
                    'taxa_observada_horizonte:Q',
                    title='Média observada do horizonte',
                    format='.2f',
                )],
            )
        )
    return alt.layer(*layers).resolve_scale(y='shared').properties(height=300)


def prediction_layer(indicators, predictions, origin, cohort, window=None):
    selected = predictions.loc[
        predictions.ano.eq(origin) & predictions.coorte.eq(cohort)
    ].copy()
    if window is not None:
        selected = selected.loc[selected.janela.eq(window)].copy()
    if not selected.divisao.eq('test').all():
        raise ValueError('A camada preditiva deve conter somente observações de teste.')
    predicted_rate_columns = [
        column for column in selected.columns
        if re.fullmatch(r'taxa_media_\d+_anos_prevista', column)
    ]
    target_rate_columns = [
        column for column in selected.columns
        if re.fullmatch(r'alvo_taxa_media_\d+_anos', column)
    ]
    predicted_percent_columns = [
        column for column in selected.columns
        if re.fullmatch(r'previsao_variacao_media_\d+_anos_percentual', column)
    ]
    target_percent_columns = [
        column for column in selected.columns
        if re.fullmatch(r'alvo_variacao_media_\d+_anos_percentual', column)
    ]
    smoothed_columns = {
        'taxa_media_3_anos_atual',
        'probabilidade_alto_risco',
    }
    if (smoothed_columns.issubset(selected.columns)
            and len(predicted_rate_columns) == 1
            and len(target_rate_columns) == 1
            and len(predicted_percent_columns) == 1
            and len(target_percent_columns) == 1):
        if selected.municipio_codigo.duplicated().any():
            raise ValueError('A camada preditiva precisa de uma janela específica para cada origem.')
        result = selected.copy()
        result['taxa_atual'] = result['taxa_media_3_anos_atual']
        result['taxa_prevista'] = result[predicted_rate_columns[0]]
        result['taxa_observada_alvo'] = result[target_rate_columns[0]]
        result['previsao_percentual'] = result[predicted_percent_columns[0]]
        result['alvo_percentual'] = result[target_percent_columns[0]]
        result['taxa_media_futura_prevista'] = result['taxa_prevista']
        result['alvo_taxa_media_futura'] = result['taxa_observada_alvo']
        result['previsao_variacao_media_futura_percentual'] = result['previsao_percentual']
        result['alvo_variacao_media_futura_percentual'] = result['alvo_percentual']
        result['valor'] = result['previsao_percentual']
        result['referencia'] = result['ano_futuro']
        result['situacao'] = np.where(
            result['valor'].notna(),
            'Previsão retrospectiva de teste',
            'Sem previsão elegível',
        )
        return result

    base = indicators.loc[indicators.ano.eq(origin),['municipio_codigo','municipio','uf','ano']].copy()
    target_column = 'alvo_log_taxa_100' if 'alvo_log_taxa_100' in selected else 'alvo_log_100'
    columns=['municipio_codigo','previsao_percentual','taxa_atual','taxa_prevista','alvo_percentual','modelo','experimento','ano_futuro','janela',target_column]
    result = base.merge(selected[columns],on='municipio_codigo',how='left',validate='one_to_one')
    result['valor']=result.previsao_percentual
    result['referencia']=result.ano_futuro
    result['situacao']=np.where(result.valor.notna(),'Previsão retrospectiva de teste','Sem previsão elegível')
    if target_column == 'alvo_log_taxa_100':
        result['taxa_observada_alvo'] = np.expm1(result[target_column] / 100)
    else:
        result['taxa_observada_alvo']=(1+result.taxa_atual)*np.exp(result[target_column]/100)-1
    return result


def future_prediction_layer(predictions, origin, cohort):
    """Prepara a camada 2024--2026 sem exigir alvo observado."""

    selected = predictions.loc[
        predictions.ano.eq(origin) & predictions.coorte.eq(cohort)
    ].copy()
    if selected.empty:
        raise ValueError('Não há previsões futuras para a origem e coorte selecionadas.')
    if not selected.divisao.eq('futuro').all():
        raise ValueError('A camada futura deve conter somente previsões não avaliadas.')
    if selected.municipio_codigo.duplicated().any():
        raise ValueError('A camada futura não pode repetir um município na mesma origem.')
    result = selected.copy()
    result['taxa_atual'] = result['taxa_media_3_anos_atual']
    result['valor'] = result['taxa_prevista']
    result['referencia'] = result['ano_futuro']
    result['previsao_percentual'] = result['previsao_percentual']
    result['situacao'] = 'Previsão futura não avaliada'
    return result


def prediction_metric_layer(frame, metric):
    """Seleciona a variável que será colorida no mapa preditivo."""

    columns = {
        'taxa': 'taxa_prevista',
        'variacao': 'previsao_percentual',
        'faixa': 'probabilidade_alto_risco_calibrada',
        'risco': (
            'probabilidade_alto_risco_calibrada'
            if 'probabilidade_alto_risco_calibrada' in frame
            else 'probabilidade_alto_risco'
        ),
        'ranking': 'percentil_ranking_taxa',
        'incerteza': 'largura_intervalo_taxa',
    }
    if metric not in columns:
        raise ValueError(f'Métrica preditiva desconhecida: {metric}')
    if metric != 'incerteza' and columns[metric] not in frame:
        raise ValueError(f'A métrica {metric} não está disponível na camada.')
    result = frame.copy()
    if metric == 'incerteza':
        lower = next(
            (column for column in result if re.fullmatch(
                r'intervalo_inferior_taxa_media_\d+_anos', column
            )),
            None,
        )
        upper = next(
            (column for column in result if re.fullmatch(
                r'intervalo_superior_taxa_media_\d+_anos', column
            )),
            None,
        )
        if lower is None or upper is None:
            raise ValueError('Intervalos de incerteza não estão disponíveis na camada.')
        result['largura_intervalo_taxa'] = result[upper] - result[lower]
    result['valor'] = result[columns[metric]]
    return result


def apply_prediction_filters(frame, risk_band='Todas', ranking_cutoff=None,
                             population_min=0):
    """Aplica filtros interpretativos sem recalcular ranking ou probabilidade."""

    result = frame.copy()
    if risk_band and risk_band != 'Todas':
        if 'faixa_risco' not in result:
            raise ValueError('A faixa de risco não está disponível na camada.')
        result = result[result.faixa_risco.eq(risk_band)]
    if ranking_cutoff is not None:
        if 'percentil_ranking_taxa' not in result:
            raise ValueError('O ranking não está disponível na camada.')
        result = result[result.percentil_ranking_taxa.ge(ranking_cutoff)]
    if population_min:
        if 'populacao' not in result:
            raise ValueError('A população não está disponível na camada.')
        result = result[result.populacao.ge(population_min)]
    return result.reset_index(drop=True)


def scale_bounds(frame, base_low, base_high, diverging=False, robust=False,
                 lower_quantile=0.01, upper_quantile=0.99):
    """Retorna limites absolutos ou robustos para a escala cromática."""

    if not robust or 'valor' not in frame:
        return float(base_low), float(base_high)
    values = pd.to_numeric(frame['valor'], errors='coerce').dropna()
    if values.empty:
        return float(base_low), float(base_high)
    lower = float(values.quantile(lower_quantile))
    upper = float(values.quantile(upper_quantile))
    if not np.isfinite(lower + upper) or lower >= upper:
        return float(base_low), float(base_high)
    if diverging:
        limit = max(abs(lower), abs(upper))
        return -limit, limit
    return lower, upper


def color(value, low, high, diverging=False, sequential_palette='teal'):
    if pd.isna(value): return [180,185,193,160]
    fraction=float(np.clip((value-low)/(high-low),0,1))
    if diverging:
        a,b,t=([49,130,189],[245,245,245],fraction*2) if fraction<=.5 else ([245,245,245],[203,62,56],(fraction-.5)*2)
    elif sequential_palette == 'yellow_red':
        a,b,t=([255,247,188],[254,153,41],fraction*2) if fraction<=.5 else ([254,153,41],[177,0,38],(fraction-.5)*2)
    elif sequential_palette == 'teal':
        a,b,t=[224,242,241],[0,77,64],fraction
    else:
        raise ValueError(f'Paleta sequencial desconhecida: {sequential_palette}')
    return [round(x+(y-x)*t) for x,y in zip(a,b)]+[220]


def map_features(geometry, frame, low, high, uf=None, diverging=False,
                 include_missing=True, sequential_palette='teal'):
    """Monta a camada GeoJSON sem alterar a tabela de origem.

    Por padrão, polígonos sem valor são preservados para manter a semântica
    histórica da função. A aplicação pode omiti-los da camada WebGL para
    reduzir o payload do navegador; a ausência continua disponível na tabela.
    """
    rows=frame.set_index('municipio_codigo').to_dict('index')
    features=[]
    for feature in geometry['features']:
        code=feature['properties']['codarea']
        if uf and UF.get(code[:2])!=uf: continue
        row=rows.get(code,{})
        value=row.get('valor',np.nan)
        if not include_missing and (not row or pd.isna(value)):
            continue
        reference=row.get('referencia',np.nan)
        props={'codigo':code,'municipio':row.get('municipio',f'Município {code}'),
            'uf':UF.get(code[:2],code[:2]),'valor_texto':f'{value:,.2f}' if pd.notna(value) else 'Sem dados',
            'referencia_texto':str(int(reference)) if pd.notna(reference) else '—',
            'situacao':row.get('situacao','Sem dados na base'),
            'faixa_risco':row.get('faixa_risco','—'),
            'ranking_texto':str(int(row['ranking_taxa_futura'])) if pd.notna(row.get('ranking_taxa_futura')) else '—',
            'intervalo_texto':(
                f"{row['intervalo_inferior_taxa_media_2_anos']:.2f}–{row['intervalo_superior_taxa_media_2_anos']:.2f}"
                if pd.notna(row.get('intervalo_inferior_taxa_media_2_anos'))
                and pd.notna(row.get('intervalo_superior_taxa_media_2_anos')) else '—'
            ),
            'cor':color(value,low,high,diverging,sequential_palette)}
        features.append({'type':'Feature','geometry':feature['geometry'],'properties':props})
    return {'type':'FeatureCollection','features':features}


def _geometry_points(coordinates):
    if isinstance(coordinates, (list, tuple)) and coordinates and isinstance(coordinates[0], (int, float)):
        yield coordinates
    elif isinstance(coordinates, (list, tuple)):
        for child in coordinates:
            yield from _geometry_points(child)


def _centroid(geometry):
    points=list(_geometry_points(geometry['coordinates']))
    if not points:
        return None
    return [
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    ]


def map_positions(geometry, frame, uf=None):
    """Retorna os centróides dos municípios presentes no recorte atual."""

    codes = set(frame.municipio_codigo.dropna().astype('string'))
    positions = []
    for feature in geometry['features']:
        code = feature['properties']['codarea']
        if code not in codes:
            continue
        if uf and UF.get(code[:2]) != uf:
            continue
        position = _centroid(feature['geometry'])
        if position is not None:
            positions.append({'codigo': code, 'position': position})
    return positions


def map_points(geometry, frame, low, high, uf=None, diverging=False,
               sequential_palette='teal'):
    """Monta pontos municipais leves para a visão nacional do mapa."""
    rows=frame.set_index('municipio_codigo').to_dict('index')
    points=[]
    for item in map_positions(geometry, frame, uf):
        code=item['codigo']
        row=rows.get(code,{})
        value=row.get('valor',np.nan)
        if not row or pd.isna(value):
            continue
        reference=row.get('referencia',np.nan)
        points.append({
            'position': item['position'],
            'codigo': code,
            'municipio': row.get('municipio',f'Município {code}'),
            'uf': UF.get(code[:2],code[:2]),
            'valor_texto': f'{value:,.2f}',
            'referencia_texto': str(int(reference)) if pd.notna(reference) else '—',
            'situacao': row.get('situacao','Sem dados na base'),
            'faixa_risco':row.get('faixa_risco','—'),
            'ranking_texto':str(int(row['ranking_taxa_futura'])) if pd.notna(row.get('ranking_taxa_futura')) else '—',
            'intervalo_texto':(
                f"{row['intervalo_inferior_taxa_media_2_anos']:.2f}–{row['intervalo_superior_taxa_media_2_anos']:.2f}"
                if pd.notna(row.get('intervalo_inferior_taxa_media_2_anos'))
                and pd.notna(row.get('intervalo_superior_taxa_media_2_anos')) else '—'
            ),
            'cor': color(value,low,high,diverging,sequential_palette),
        })
    return points
