# Previsão futura: origem 2024 e horizonte 2024–2026

## Objetivo

Esta etapa transforma o experimento retrospectivo final em uma previsão futura. A origem é o ano de 2024 e o horizonte é de dois anos: 2025 e 2026. Como os valores observados desses anos ainda não estão disponíveis no painel utilizado, a saída é identificada como **previsão futura não avaliada**.

O Atlas da Violência continua sendo a fonte principal das taxas municipais. A população de 2024 vem do arquivo SIDRA local e é usada para a coorte populacional e para a contagem proxy.

## Dados e rastreabilidade

Foram utilizados:

- `data/raw/Taxa de homicídios registrados_municípios.csv`: taxas municipais do Atlas até 2024;
- `data/raw/populacao_municipal_ibge_2001_2021.csv`: série histórica de população;
- `data/raw/sidra_populacao_2022.json`: população do Censo 2022;
- `data/raw/sidra_populacao_2024.json`: estimativa populacional de 2024;
- `data/raw/municipios_ibge.json`: resolução dos nomes e códigos municipais.

Os hashes SHA-256, tamanhos dos arquivos e a configuração completa estão em `data/processed/v11/config_futuro.json`. As fontes, URLs e limitações gerais estão detalhadas em [`fontes-e-rastreabilidade.md`](fontes-e-rastreabilidade.md).

## Construção da origem futura

Para cada município, a origem de 2024 só é criada quando existem taxas consecutivas para 2020, 2021, 2022, 2023 e 2024. Assim, não são preenchidas ausências do Atlas com zero e não são utilizados dados posteriores à origem.

As 17 features são as mesmas do modelo final: taxa corrente, defasagens, médias, dispersões, mediana, amplitude, tendências, número de aumentos, variação logarítmica e logaritmo da população. A população de 2024 é selecionada por município e não recalcula a taxa publicada pelo Atlas.

## Treinamento sem vazamento temporal

O modelo futuro é treinado somente em linhas históricas que possuem o alvo observado. Com horizonte de dois anos, o último ano de origem que possui alvo completo é 2022, pois ele permite observar 2023 e 2024. Portanto, a taxa de 2024 é usada como feature da previsão futura, mas não como parte de um alvo 2025–2026.

Foram reutilizados os hiperparâmetros escolhidos pela validação temporal do MVP final. Para cada coorte, o algoritmo selecionado na janela histórica mais recente foi treinado novamente com todas as observações históricas elegíveis:

| Coorte | Modelo de taxa | Modelo de risco | Observações históricas | Municípios previstos |
|---|---|---:|---:|---:|
| População ≥ 50 mil | HistGradientBoosting | Regressão logística | 11.621 | 664 |
| Todos os portes elegíveis | HistGradientBoosting | HistGradientBoosting | 38.930 | 2.565 |

O risco é definido como a probabilidade de pertencer ao quartil superior da taxa futura histórica. A calibração da probabilidade foi ajustada na janela temporal 2017–2018. O intervalo conformal de 90% também foi calculado usando resíduos dessa janela, sem utilizar qualquer observação de 2025 ou 2026.

## Artefatos gerados

- `data/processed/v11/previsoes_futuras_2024_2026.csv`: saída municipal para o mapa;
- `data/processed/v11/config_futuro.json`: parâmetros, fontes, hashes e limitações;
- `models/v11/futuro_2024_2026/`: modelos finais por coorte.

Cada linha contém a taxa média futura prevista, variação em relação à média histórica recente, probabilidade calibrada de alto risco, faixa de risco, ranking relativo, intervalo inferior e superior, população, exposição de dois anos e contagem proxy.

## Resultados descritivos preliminares

Os resultados gerados em 28 de setembro de 2026 possuem 3.229 linhas. As coortes não devem ser somadas: `todos` contém também os municípios da coorte de 50 mil habitantes.

| Coorte | Taxa prevista média | Mediana | Intervalo médio | Contagem proxy média por município |
|---|---:|---:|---:|---:|
| População ≥ 50 mil | 20,242 | 18,420 | 10,850–37,080 | 80,147 |
| Todos os portes elegíveis | 24,728 | 23,498 | 11,325–52,707 | 28,614 |

A contagem proxy média da tabela é uma média municipal, não a soma da coorte. A soma da contagem proxy é 53.217,48 para a coorte principal e 73.395,75 para todos os municípios, mas esses números não representam homicídios observados nem devem ser interpretados como contagens oficiais.

Distribuição das faixas de risco:

| Coorte | Baixo | Moderado | Alto | Muito alto |
|---|---:|---:|---:|---:|
| População ≥ 50 mil | 612 | 32 | 17 | 3 |
| Todos os portes elegíveis | 2.207 | 240 | 111 | 7 |

Esses valores são uma primeira fotografia da saída do modelo. Eles não demonstram que a violência irá aumentar nos municípios classificados como alto ou muito alto. A classificação indica semelhança com o quartil superior do alvo histórico, sob as hipóteses do modelo.

## Aplicação no mapa

Ao iniciar a aplicação, a opção **Previsão futura 2024→2026** apresenta exclusivamente o arquivo futuro. Ela pode ser visualizada por:

- taxa média futura prevista;
- faixa de risco;
- probabilidade calibrada de alto risco;
- ranking relativo;
- largura do intervalo de incerteza;
- variação percentual prevista.

A camada futura é mantida separada das janelas retrospectivas de teste. Assim, o mapa não apresenta os valores observados de 2025 ou 2026 como se já fossem conhecidos.

## Validação funcional

Em 28 de setembro de 2026, foi feita uma verificação funcional local dos
artefatos e da aplicação:

- as duas coortes carregaram, com 664 e 2.565 municípios;
- taxa, faixa, risco, ranking, incerteza e variação produziram valores para
  todas as linhas das duas coortes;
- a visão nacional gerou pontos municipais e a visão por UF gerou polígonos;
- a camada retrospectiva de teste continuou carregando separadamente;
- o Streamlit iniciou em modo headless sem erro.

Essa verificação confirma o funcionamento do fluxo de dados e da interface,
mas não substitui a validação estatística futura contra os valores observados.

Execute:

```bash
source .venv/bin/activate
python -m src.forecast_futura
streamlit run app/main.py
```

## Limitações

1. A previsão 2024–2026 ainda não possui métrica de erro real.
2. A validação futura só será possível quando houver valores oficiais observados para 2025 e 2026.
3. A contagem é uma conversão da taxa prevista para uma exposição populacional de dois anos; não é numerador oficial do Atlas nem do SIM.
4. O intervalo conformal é uma estimativa de cobertura baseada no histórico de validação, não uma garantia individual.
5. O resultado apoia análise e planejamento acadêmico; não deve ser usado como alerta operacional automático ou como justificativa isolada para ações contra um município.
