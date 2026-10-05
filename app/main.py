"""Execute: streamlit run app/main.py."""
import sys
from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.data import (
    LAYERS,
    UF,
    apply_prediction_filters,
    build_temporal_chart,
    build_prediction_temporal_chart,
    descriptive_layer,
    map_positions,
    load_data,
    map_features,
    map_points,
    prediction_layer,
    prediction_metric_layer,
    scale_bounds,
    future_prediction_layer,
    value_definition,
)

st.set_page_config(page_title='Violência letal | Indicadores municipais',layout='wide')


@st.cache_data
def cached_data():
    return load_data()


def main():
    st.title('Violência letal e indicadores municipais')
    st.caption('Dados públicos · exploração histórica · MVP acadêmico')
    try:
        data=cached_data()
    except (FileNotFoundError,ValueError) as exc:
        st.error(f'Não foi possível carregar os artefatos: {exc}')
        st.info('Prepare os dados com python -m src.experiment_final e mantenha a malha municipal em data/geo/.')
        st.stop()
    indicators=data['indicadores_mapa']
    predictions=data['previsoes_mapa']
    mode=st.sidebar.radio('Visualização',[
        'Indicadores observados', 'Previsões retrospectivas', 'Previsão futura 2024→2026'
    ],key='mode')
    uf=st.sidebar.selectbox('Estado',['Todos']+sorted(indicators.uf.dropna().unique()),key='uf')
    chosen_uf=None if uf=='Todos' else uf
    if mode=='Indicadores observados':
        layer=st.sidebar.selectbox('Indicador',list(LAYERS),key='layer')
        field,unit,low,high=LAYERS[layer]
        year=st.sidebar.selectbox('Ano',sorted(indicators.ano.unique(),reverse=True),key='year')
        exact=st.sidebar.checkbox('Somente observações do ano',key='exact')
        frame=descriptive_layer(indicators,year,field,exact)
        st.subheader(f'{layer} · {year}')
        st.caption(f'Unidade: {unit}. Municípios sem valor ficam fora da camada visual e continuam disponíveis na tabela; ausência não equivale a zero. Referências censitárias anteriores aparecem identificadas quando a opção “somente observações do ano” está desligada.')
        diverging=False
        sequential_palette='yellow_red' if field == 'taxa_homicidios' else 'teal'
    elif mode=='Previsões retrospectivas':
        cohort=st.sidebar.selectbox('Coorte',['principal_50mil','todos'],
            format_func=lambda v:'População ≥ 50 mil' if v=='principal_50mil' else 'Todos os portes elegíveis',key='cohort')
        windows=sorted(predictions.loc[predictions.coorte.eq(cohort),'janela'].dropna().unique(),reverse=True)
        window=st.sidebar.selectbox('Janela de teste',windows,key='prediction_window')
        years=sorted(predictions.loc[predictions.coorte.eq(cohort)&predictions.janela.eq(window),'ano'].unique(),reverse=True)
        year=st.sidebar.selectbox('Ano de origem',years,key='origin')
        frame=prediction_layer(indicators,predictions,year,cohort,window)
        metric=st.sidebar.selectbox(
            'Variável no mapa',
            ['taxa','faixa','risco','ranking','incerteza','variacao'],
            format_func=lambda v: {
                'taxa':'Taxa média futura prevista',
                'faixa':'Faixa de risco',
                'ranking':'Ranking relativo do município',
                'incerteza':'Largura do intervalo de previsão',
                'variacao':'Variação percentual prevista',
                'risco':'Probabilidade calibrada de alto risco',
            }[v],
            key='prediction_metric',
        )
        frame=prediction_metric_layer(frame,metric)
        low,high,diverging = {
            'taxa': (0,100,False),
            'faixa': (0,1,False),
            'variacao': (-100,100,True),
            'risco': (0,1,False),
            'ranking': (0,1,False),
            'incerteza': (0,100,False),
        }[metric]
        sequential_palette='yellow_red'
        if metric == 'incerteza' and frame['valor'].notna().any():
            high = max(1.0, float(frame['valor'].quantile(0.95)))
        horizon=int(data['config'].get('horizonte_anos',2))
        st.subheader(f'Previsão retrospectiva · origem {year} → média futura {year+1}–{year+horizon}')
        st.info(f'Avaliação retrospectiva em dados de teste. A taxa prevista é a média dos {horizon} anos seguintes; a probabilidade de risco foi calibrada usando somente a validação. Os intervalos de 90% são conformais e estimados pelos resíduos da validação; as faixas e o ranking são relativos à coorte e à janela selecionadas.')
        selection=data['selecao_validacao'].query('coorte == @cohort and janela == @window').iloc[0]
        metrics=data['metricas_janelas']
        result=metrics[(metrics.coorte==cohort)&(metrics.janela==window)&(metrics.modelo==selection.modelo)&(metrics.experimento==selection.experimento)&metrics.divisao.eq('test')].iloc[0]
        risk_selection=data['selecao_risco_validacao'].query('coorte == @cohort and janela == @window').iloc[0]
        risk_metrics=data['metricas_risco']
        risk_result=risk_metrics[(risk_metrics.coorte==cohort)&(risk_metrics.janela==window)&(risk_metrics.modelo==risk_selection.modelo)&(risk_metrics.experimento==risk_selection.experimento)&risk_metrics.divisao.eq('test')].iloc[0]
        st.caption(f'Candidatos escolhidos na validação: regressão {selection.modelo}; classificação {risk_selection.modelo}. Janela de teste {window}, n = {int(result.n):,}.')
        metric_columns=st.columns(5)
        metric_columns[0].metric('MAE da taxa',f'{result.mae_taxa:.2f}')
        metric_columns[1].metric('MAE percentual',f'{result.mae_percentual:.2f} p.p.')
        balanced_column = 'balanced_accuracy_calibrado' if 'balanced_accuracy_calibrado' in risk_result else 'balanced_accuracy'
        roc_column = 'roc_auc_calibrado' if 'roc_auc_calibrado' in risk_result else 'roc_auc'
        metric_columns[2].metric('Balanced accuracy',f'{risk_result[balanced_column]:.3f}')
        brier_column = 'brier_score_calibrado' if 'brier_score_calibrado' in risk_result else 'brier_score'
        metric_columns[3].metric('Brier Score',f'{risk_result[brier_column]:.3f}')
        metric_columns[4].metric('ROC-AUC',f'{risk_result[roc_column]:.3f}')
    else:
        future_predictions=data['previsoes_futuras']
        if future_predictions.empty:
            st.error('A camada futura ainda não foi gerada.')
            st.info('Execute python -m src.forecast_futura após preparar os artefatos finais.')
            st.stop()
        cohort=st.sidebar.selectbox('Coorte',['principal_50mil','todos'],
            format_func=lambda v:'População ≥ 50 mil' if v=='principal_50mil' else 'Todos os portes elegíveis',key='future_cohort')
        years=sorted(future_predictions.loc[future_predictions.coorte.eq(cohort),'ano'].unique(),reverse=True)
        year=st.sidebar.selectbox('Ano de origem',years,key='future_origin')
        frame=future_prediction_layer(future_predictions,year,cohort)
        metric=st.sidebar.selectbox(
            'Variável no mapa',
            ['taxa','faixa','risco','ranking','incerteza','variacao'],
            format_func=lambda v: {
                'taxa':'Taxa média futura prevista',
                'faixa':'Faixa de risco',
                'ranking':'Ranking relativo do município',
                'incerteza':'Largura do intervalo de previsão',
                'variacao':'Variação percentual prevista',
                'risco':'Probabilidade calibrada de alto risco',
            }[v],
            key='future_prediction_metric',
        )
        frame=prediction_metric_layer(frame,metric)
        low,high,diverging = {
            'taxa': (0,100,False),
            'faixa': (0,1,False),
            'variacao': (-100,100,True),
            'risco': (0,1,False),
            'ranking': (0,1,False),
            'incerteza': (0,100,False),
        }[metric]
        sequential_palette='yellow_red'
        if metric == 'incerteza' and frame['valor'].notna().any():
            high = max(1.0, float(frame['valor'].quantile(0.95)))
        horizon=int(data['config'].get('horizonte_anos',2))
        st.subheader(f'Previsão futura · origem {year} → {year+1}–{year+horizon}')
        st.info('Esta camada é uma projeção 2024→2026 e ainda não foi avaliada contra valores observados. A taxa é uma média prevista para os dois anos; as faixas, o ranking e o intervalo de 90% são instrumentos de interpretação, não alertas operacionais.')
    if mode == 'Indicadores observados':
        value_label, value_explanation = value_definition('observed', field)
    else:
        value_label, value_explanation = value_definition('prediction', metric)
    st.sidebar.caption(f'{value_label}: {value_explanation}')
    if mode != 'Indicadores observados':
        risk_options=['Todas']+sorted(frame.faixa_risco.dropna().unique().tolist())
        risk_band=st.sidebar.selectbox('Faixa de risco',risk_options,key='risk_band')
        rank_options={
            'Todos': None,
            '10% superiores': .90,
            '25% superiores': .75,
            '50% superiores': .50,
        }
        rank_label=st.sidebar.selectbox('Filtro de ranking',list(rank_options),key='rank_filter')
        population_options=[0,10_000,50_000,200_000,500_000]
        population_min=st.sidebar.selectbox(
            'População mínima',population_options,
            format_func=lambda value:'Todos os municípios' if value==0 else f'{value:,} habitantes',
            key='prediction_population_min',
        )
        frame=apply_prediction_filters(
            frame,risk_band,rank_options[rank_label],population_min
        )
    else:
        population_min=st.sidebar.selectbox(
            'População mínima',[0,10_000,50_000,200_000,500_000],
            format_func=lambda value:'Todos os municípios' if value==0 else f'{value:,} habitantes',
            key='observed_population_min',
        )
        if population_min and 'populacao' in frame:
            frame=frame[frame.populacao.ge(population_min)].copy()
    if chosen_uf:
        frame=frame[frame.uf.eq(chosen_uf)].copy()
    if frame.empty:
        st.warning('Nenhum município atende aos filtros selecionados.')
        st.stop()
    map_layer_mode=st.sidebar.selectbox(
        'Camada geográfica',['Automática','Pontos','Polígonos'],
        help='Automática usa pontos no Brasil e polígonos quando uma UF é selecionada.',
        key='map_layer_mode',
    )
    scale_mode=st.sidebar.selectbox(
        'Escala de cores',['Fixa','Robusta (1º–99º percentil)'],
        help='A escala robusta reduz o efeito visual de valores extremos, sem alterar os dados.',
        key='scale_mode',
    )
    low,high=scale_bounds(
        frame,low,high,diverging,
        robust=scale_mode.startswith('Robusta'),
    )
    geometry_codes={
        f['properties']['codarea']
        for f in data['geometry']['features']
        if chosen_uf is None or UF.get(f['properties']['codarea'][:2]) == chosen_uf
    }
    shown=frame[frame.municipio_codigo.isin(geometry_codes)]
    a,b,c=st.columns(3)
    a.metric('Municípios na malha',len(geometry_codes))
    b.metric('Com valor no mapa',int(shown.valor.notna().sum()))
    c.metric('Sem valor no mapa',len(geometry_codes)-int(shown.valor.notna().sum()))
    unmatched=frame[~frame.municipio_codigo.isin(geometry_codes)]
    if len(unmatched):
        st.warning(f'{len(unmatched)} município(s) da tabela não possuem polígono na malha de 2022; permanecem disponíveis na tabela.')
    tooltip={'text':(
        '{municipio} ({uf})\n'
        'Código: {codigo}\n'
        f'{value_label}: {{valor_texto}}\n'
        f'Interpretação: {value_explanation}\n'
        'Referência: {referencia_texto}\n'
        '{situacao}'
    )}
    use_points=map_layer_mode=='Pontos' or (
        map_layer_mode=='Automática' and chosen_uf is None
    )
    if use_points:
        points=map_points(data['geometry_map'],frame,low,high,chosen_uf,
            diverging,sequential_palette)
        map_layer=pdk.Layer('ScatterplotLayer',points,id='municipios',pickable=True,
            filled=True,get_position='position',get_fill_color='cor',get_radius=10000,
            radius_min_pixels=1.5,radius_max_pixels=12)
    else:
        geo=map_features(data['geometry_map'],frame,low,high,chosen_uf,diverging,
            include_missing=False,sequential_palette=sequential_palette)
        map_layer=pdk.Layer('GeoJsonLayer',geo,id='municipios',pickable=True,filled=True,
            stroked=True,get_fill_color='properties.cor',get_line_color=[255,255,255,90],
            line_width_min_pixels=.3)
    positions=map_positions(data['geometry_map'],frame,chosen_uf)
    if positions:
        fitted=pdk.data_utils.compute_view(
            [item['position'] for item in positions],view_proportion=1
        )
        view_state=pdk.ViewState(
            latitude=fitted.latitude,
            longitude=fitted.longitude,
            zoom=min(max(float(fitted.zoom),2.5),10),
        )
    else:
        view_state=pdk.ViewState(latitude=-14.2,longitude=-52.0,zoom=3.2)
    deck=pdk.Deck(map_provider='carto',map_style='light',
        initial_view_state=view_state,
        parameters={'clearColor':[0.96,0.97,0.98,1]},layers=[map_layer],tooltip=tooltip)
    st.pydeck_chart(deck,height=560)
    if use_points:
        st.caption('Camada de pontos municipais. A viewport foi ajustada automaticamente ao recorte atual.')
    elif chosen_uf is None:
        st.caption('Camada de polígonos municipais. A viewport foi ajustada automaticamente ao recorte atual.')
    elif mode in ('Previsões retrospectivas','Previsão futura 2024→2026') and metric in ('risco','faixa'):
        st.caption('Escala de probabilidade; as faixas não representam aumento percentual.')
    elif mode in ('Previsões retrospectivas','Previsão futura 2024→2026') and metric=='ranking':
        st.caption('Escala relativa de 0 a 1: valores maiores indicam posição mais alta no ranking da taxa futura prevista dentro da coorte, janela e ano de origem.')
    elif mode in ('Previsões retrospectivas','Previsão futura 2024→2026') and metric=='incerteza':
        st.caption('A cor representa a largura do intervalo conformal da taxa futura prevista. Intervalos maiores indicam maior incerteza relativa; não são garantia individual.')
    elif diverging:
        st.caption(f'Escala {scale_mode.lower()}: azul = redução, branco = zero, vermelho = aumento; limites {low:.1f} a {high:.1f}.')
    elif sequential_palette == 'yellow_red':
        st.caption(f'Escala {scale_mode.lower()}: amarelo claro = menor valor, vermelho escuro = maior valor; limites {low:,.2f} a {high:,.2f}.')
    else:
        st.caption(f'Escala {scale_mode.lower()}: verde claro → verde escuro, de {low:,.2f} a {high:,.2f}. Valores fora dos limites saturam a cor.')
    options=frame[['municipio_codigo','municipio','uf']].sort_values(['municipio','uf'])
    names={r.municipio_codigo:f'{r.municipio} — {r.uf}' for r in options.itertuples()}
    city=st.selectbox('Ficha municipal',options.municipio_codigo.tolist(),format_func=lambda code:names[code],key='city')
    if city:
        row=frame[frame.municipio_codigo.eq(city)].iloc[0]
        st.write(f'**{names[city]}** · {row.situacao}')
        if mode=='Indicadores observados':
            series=indicators[indicators.municipio_codigo.eq(city)].copy()
            ref=f'{field}_ano_referencia'
            if ref in series:
                series=series[series.ano.eq(series[ref])]
            # Pontos censitários: evita sugerir medições nos anos intermediários.
            st.altair_chart(
                build_temporal_chart(series, field, layer, unit),
                use_container_width=True,
            )
            details=row[['valor','referencia','situacao']].map(
                lambda value: '—' if pd.isna(value) else str(value)
            ).to_frame('Informação')
            st.dataframe(details)
        else:
            series = indicators[indicators.municipio_codigo.eq(city)].copy()
            st.altair_chart(
                build_prediction_temporal_chart(
                    series,
                    row,
                    'Taxa de homicídios',
                    'por 100 mil habitantes',
                ),
                use_container_width=True,
            )
            st.caption('Linha cinza: taxa observada. Linha laranja: média prevista para todo o horizonte. Faixa e linhas laranjas tracejadas: intervalo de incerteza de 90%. O losango azul aparece somente na avaliação retrospectiva e representa a média observada do horizonte.')
            columns = [
                'taxa_media_3_anos_atual', 'taxa_prevista',
                'alvo_taxa_media_futura',
                'previsao_percentual', 'alvo_percentual',
                'intervalo_inferior_taxa_media_2_anos',
                'intervalo_superior_taxa_media_2_anos',
                'ranking_taxa_futura', 'percentil_ranking_taxa', 'faixa_risco',
                'probabilidade_alto_risco', 'probabilidade_alto_risco_calibrada',
                'risco_alto_previsto', 'risco_alto_previsto_calibrado',
                'limiar_taxa_alto_risco', 'modelo_regressao',
                'modelo_classificacao', 'janela',
            ]
            details=row[[column for column in columns if column in row]].map(
                lambda value: '—' if pd.isna(value) else str(value)
            ).to_frame('Informação')
            st.dataframe(details)
    with st.expander('Tabela e download'):
        if mode!='Indicadores observados':
            columns=['municipio_codigo','municipio','uf','ano','ano_futuro','valor',
                'taxa_media_3_anos_atual','taxa_prevista',
                'previsao_percentual','probabilidade_alto_risco_calibrada',
                'intervalo_inferior_taxa_media_2_anos',
                'intervalo_superior_taxa_media_2_anos',
                'ranking_taxa_futura','percentil_ranking_taxa','faixa_risco',
                'risco_alto_previsto_calibrado','coorte','janela','modelo_regressao',
                'modelo_classificacao','situacao']
        else:
            columns=['municipio_codigo','municipio','uf','ano','valor','referencia','situacao']
        columns=[column for column in columns if column in frame]
        table=frame[columns].sort_values(['uf','municipio'])
        st.dataframe(table,hide_index=True)
        st.download_button('Baixar tabela filtrada',table.to_csv(index=False).encode('utf-8-sig'),
            file_name=f"{'previsoes_futuras_2024_2026' if mode=='Previsão futura 2024→2026' else 'previsoes_retrospectivas' if mode=='Previsões retrospectivas' else 'indicadores'}_{year}.csv",mime='text/csv')
    with st.expander('Fontes e limites da análise'):
        st.markdown('Taxas: [Atlas da Violência](https://www.ipea.gov.br/atlasviolencia/tema/1/). População, cadastro municipal e malha: [IBGE](https://www.ibge.gov.br/). Indicadores sociais observados: [Ipeadata](https://www.ipeadata.gov.br/).')
        st.write('Ausências no CSV de taxas não foram tratadas como zero. GINI, IDHM e PIND terminam em 2010; analfabetismo tem 2022. A população reúne estimativas e censos e não recalcula a taxa do Atlas. Não há população anual observada para 2023 nesta base.')
        st.write('A malha de 2022 é fixa e não reconstrói limites históricos. As previsões retrospectivas dependem de séries completas, usam somente observações de teste e não cobrem todos os municípios. A taxa futura é uma média de dois anos; a variação percentual é derivada e continua sujeita a grande erro em taxas-base baixas. O histórico ampliado usa cinco anos anteriores à origem. As faixas de risco são baseadas em probabilidade calibrada e o ranking é relativo à coorte e à janela. O Poisson é uma análise de sensibilidade com contagem proxy derivada da taxa do Atlas, não uma contagem oficial. O mapa é retrospectivo e não constitui alerta operacional.')


if __name__=='__main__':
    main()
