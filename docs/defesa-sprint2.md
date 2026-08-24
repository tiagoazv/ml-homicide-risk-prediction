# Sprint 2 — MVP analítico

Este documento consolida a metodologia, os resultados, as decisões e as limitações da Sprint 2. Ele serve como registro técnico do projeto e como base inicial para as seções de metodologia, resultados e limitações do artigo científico.

## 1. Resumo executivo

Na Sprint 2, o projeto deixou de estar apenas na etapa exploratória e passou a ter um MVP analítico reproduzível. Foi construída uma base município-ano, foram criadas variáveis históricas e uma variável-alvo contínua para prever a variação da taxa de homicídios no ano seguinte.

Também foi integrada a população municipal do IBGE. Como as taxas de municípios pequenos sofrem grandes oscilações quando poucos eventos alteram o numerador, foi definida uma coorte principal com população mínima de 50 mil habitantes no ano de origem. A base completa foi mantida para análise de sensibilidade.

Foram comparados dois baselines, uma regressão Ridge e um Random Forest. No conjunto de teste, o Random Forest apresentou o melhor desempenho na coorte principal, com MAE logarítmico de 44,48 e R² de 0,515, superando o baseline de variação zero, que apresentou MAE de 70,20.

O resultado mostra sinal preditivo inicial, mas ainda não representa uma probabilidade operacional de risco nem uma relação causal. O MVP precisa ser ampliado com mais variáveis, análise de incerteza, validação regional e avaliação ética antes de ser utilizado em uma aplicação pública.

## 2. Problema de pesquisa

### Problema

É possível utilizar dados públicos históricos para prever quais municípios brasileiros terão aumento na taxa de homicídios no ano seguinte?

### Pergunta de pesquisa

Em que medida modelos de Machine Learning conseguem prever a variação da taxa de homicídios em municípios brasileiros, considerando o histórico da própria taxa e características populacionais?

### Objetivo geral

Desenvolver e avaliar um modelo de regressão capaz de prever a variação futura da taxa municipal de homicídios, produzindo uma saída contínua que possa posteriormente ser apresentada em um mapa interativo.

### Objetivos específicos da Sprint 2

- Definir o contrato da base analítica.
- Transformar os dados brutos em um painel município-ano.
- Criar uma variável-alvo de um ano à frente.
- Criar variáveis históricas sem vazamento temporal.
- Integrar população municipal e definir uma coorte mais estável.
- Comparar baselines e modelos de regressão.
- Avaliar o desempenho com uma divisão temporal.
- Registrar resultados, limitações e decisões metodológicas.

## 3. Dados utilizados

### 3.1 Taxas de homicídios

O arquivo `data/raw/taxa_homicidios_2024.csv` contém uma observação por município e ano, com as colunas originais `cod`, `nome`, `período` e `valor`.

Características da extração:

- 164.777 linhas;
- 5.561 municípios;
- período de 1989 a 2022;
- taxa de homicídios por 100 mil habitantes;
- ausência estrutural de anos entre 1996 e 1999;
- 1995 com cobertura parcial.

A procedência, o hash e as limitações estão registrados em `data/raw/README.md`. O espelho utilizado para o CSV não substitui a verificação da metodologia original do Atlas da Violência.

### 3.2 População municipal

O arquivo `data/raw/populacao_municipal_ibge_2001_2021.csv` consolida extrações municipais do SIDRA/IBGE:

- Tabela 6579: população residente estimada;
- Tabela 579: Contagem da População de 2007;
- Tabela 202: Censo Demográfico de 2010;
- 116.767 observações válidas entre 2001 e 2021.

A população é utilizada como variável auxiliar e para classificar o porte municipal. Ela não é usada para recalcular a taxa do Atlas, porque o CSV da taxa não fornece o denominador original e o Atlas possui metodologia populacional própria para parte do período.

#### Compatibilidade entre taxa e população

| Aspecto | Taxa do Atlas | População integrada |
| --- | --- | --- |
| Unidade territorial | Município | Município |
| Chave utilizada | Código IBGE | Código IBGE com sete dígitos |
| Unidade temporal | Ano | Ano |
| Unidade da taxa | Homicídios por 100 mil habitantes | Não é recalculada pelo projeto |
| Fonte populacional | Metodologia própria do Atlas | IBGE/SIDRA: estimativas, Contagem de 2007 e Censo de 2010 |

As fontes são compatíveis em unidade territorial, ano e conceito geral de população residente, mas não são necessariamente idênticas ao denominador utilizado pelo Atlas. A população integrada deve ser descrita como feature auxiliar, e não como substituição metodológica do denominador original.

## 4. Unidade de análise e coorte

A unidade de análise é o par município-ano. A chave é:

`municipio_codigo + ano`

O código municipal é armazenado como texto com sete dígitos para preservar a identificação IBGE.

### Coorte principal

Foi aplicado o critério:

`populacao >= 50.000`

O filtro usa apenas a população do ano de origem da previsão. Não utiliza população futura.

Resultado da coorte principal:

- 12.324 observações;
- 687 municípios ao longo do período;
- 8.963 observações de treinamento;
- 2.004 observações de validação;
- 1.357 observações de teste.

A base completa permanece disponível como sensibilidade:

- 111.156 observações;
- 5.561 municípios;
- 83.351 observações de treinamento;
- 16.683 observações de validação;
- 11.122 observações de teste.

O filtro melhora a estabilidade estatística das taxas, mas reduz a representatividade do estudo. Portanto, os resultados da coorte principal não devem ser generalizados automaticamente para municípios pequenos.

## 5. Preparação dos dados

O pipeline executa as seguintes etapas:

1. leitura do CSV de taxas;
2. padronização dos nomes das colunas;
3. conversão de tipos;
4. preenchimento do código IBGE para sete dígitos;
5. verificação de valores ausentes, taxas negativas e duplicidades;
6. ordenação por município e ano;
7. criação das janelas históricas;
8. integração da população;
9. aplicação da coorte principal;
10. divisão temporal;
11. treinamento, avaliação e salvamento dos artefatos.

As janelas são aceitas somente quando os anos são consecutivos. Isso impede que o intervalo sem dados de 1996–1999 seja interpretado como uma variação anual normal.

## 6. Variável-alvo

### Alvo primário

O alvo utilizado no treinamento é:

```text
target_variacao_log_100 =
100 × [log(1 + taxa(t+1)) − log(1 + taxa(t))]
```

Essa transformação foi escolhida porque:

- permite lidar com taxa atual igual a zero;
- reduz a influência de variações percentuais extremamente grandes;
- representa aumentos e quedas em uma escala mais equilibrada;
- produz uma variável contínua adequada para regressão.

### Variação percentual bruta

Para interpretação, também é calculada:

```text
target_variacao_percentual =
100 × [taxa(t+1) − taxa(t)] / taxa(t)
```

Essa medida não é definida quando a taxa atual é zero. Por isso, as métricas percentuais são calculadas somente nos casos válidos e podem ser bastante influenciadas por outliers.

### Interpretação correta

Um MAE de 44,48 na escala logarítmica não significa erro médio de 44,48 pontos percentuais. A métrica principal está na escala transformada. A conversão para uma porcentagem estabilizada é feita por:

```text
percentual_estabilizado = 100 × [exp(previsao_log_100 / 100) − 1]
```

### Distribuição observada na coorte principal

As estatísticas abaixo ajudam a interpretar a escala do problema:

| Variável | Média | Mediana | Primeiro quartil | Terceiro quartil | Mínimo | Máximo |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Taxa atual | 27,78 | 22,13 | 11,45 | 39,08 | 0,00 | 174,70 |
| Média histórica de 3 anos | 27,89 | 22,33 | 12,07 | 39,19 | 0,00 | 122,46 |
| Desvio-padrão histórico | 5,34 | 4,07 | 2,30 | 6,99 | 0,00 | 55,37 |
| Variação logarítmica atual | 0,07 | -0,59 | -24,93 | 24,46 | -438,12 | 443,82 |
| Alvo logarítmico | 0,67 | -0,57 | -24,79 | 24,72 | -438,12 | 566,46 |
| Alvo percentual bruto | 44,84 | -0,77 | -23,28 | 27,99 | -100,00 | 32.480,00 |

O alvo percentual bruto possui 153 valores ausentes porque a taxa atual era zero. O máximo de 32.480% mostra por que a média percentual é sensível a outliers: uma cidade que passa de uma taxa muito baixa para uma taxa moderada pode apresentar uma porcentagem enorme, mesmo que a mudança absoluta não seja tão grande.

## 7. Variáveis preditoras

As variáveis utilizadas na coorte principal são as seguintes. Todas são calculadas com informações disponíveis no ano de origem `t`; o modelo tenta prever o ano `t+1`.

| Variável | Tipo/unidade | Como é calculada | O que representa no modelo |
| --- | --- | --- | --- |
| `ano` | Inteiro, ano | Ano da observação atual | Captura tendência temporal e mudanças gerais ao longo dos anos |
| `taxa_homicidios` | Decimal, por 100 mil habitantes | Valor observado no ano `t` | Nível mais recente da violência letal |
| `taxa_lag_1` | Decimal, por 100 mil habitantes | `taxa(t-1)` | Primeiro histórico de comparação |
| `taxa_lag_2` | Decimal, por 100 mil habitantes | `taxa(t-2)` | Segundo histórico de comparação |
| `taxa_media_3_anos` | Decimal, por 100 mil habitantes | Média de `taxa(t-2)`, `taxa(t-1)` e `taxa(t)` | Nível médio recente, reduzindo o peso de um único ano |
| `taxa_std_3_anos` | Decimal, por 100 mil habitantes | Desvio-padrão populacional das três taxas | Volatilidade histórica do município |
| `variacao_log_100_atual` | Decimal, escala logarítmica × 100 | `100 × [log1p(taxa(t)) − log1p(taxa(t-1))]` | Direção e intensidade da última mudança observada |
| `populacao` | Inteiro, habitantes | População do município no ano `t` | Tamanho do município no momento da previsão |
| `log_populacao` | Decimal, sem unidade | `log1p(populacao)` | Versão estabilizada da população para uso nos modelos |

### Por que existem duas defasagens?

Uma única taxa anterior informa apenas o passado imediato. Duas defasagens permitem observar uma sequência de três anos e calcular tendência, média e volatilidade. Por exemplo, para prever 2018, a linha de origem 2017 utiliza as taxas de 2015, 2016 e 2017, desde que os anos sejam consecutivos.

### Por que utilizar média e desvio-padrão?

A média diferencia um município que permanece estruturalmente em nível alto de outro que teve apenas um pico isolado. O desvio-padrão informa se a série é estável ou oscilante. Essas variáveis não usam o ano futuro e, portanto, não introduzem o resultado que o modelo deve prever.

### Por que utilizar `log_populacao` em vez de `populacao`?

A população varia de dezenas de milhares a milhões de habitantes. O logaritmo reduz essa diferença de escala e permite que o modelo compare crescimento relativo de porte sem deixar que os maiores municípios dominem numericamente a variável.

### O que não entra como variável preditora?

`target_variacao_log_100`, `target_variacao_percentual`, `taxa_futuro` e qualquer informação do ano `t+1` não entram no treinamento. Esses campos são usados apenas para calcular o resultado e avaliar a previsão.

Nenhuma variável utiliza o alvo ou o ano futuro. A variável `porte_populacional` é usada principalmente para estratificação dos erros, não como feature categórica do modelo.

### Colunas auxiliares do painel

Além das features e do alvo, o painel contém colunas necessárias para controle e interpretação:

| Coluna | Significado | Uso |
| --- | --- | --- |
| `ano_futuro` | Ano seguinte ao ano de origem, sempre `ano + 1` | Identifica o horizonte do alvo |
| `taxa_futuro` | Taxa observada no ano seguinte | Usada somente para construir e avaliar o alvo |
| `porte_populacional` | Faixa de população: 50–200 mil, 200–500 mil ou acima de 500 mil na coorte principal | Estratifica os erros por tamanho municipal |
| `populacao_observada` | Indica se a população veio diretamente da fonte (`True`) ou foi carregada da última observação anterior (`False`) | Permite auditar a qualidade da integração |
| `populacao_fonte` | Identifica `observada` ou `ultima_observacao_anterior` | Documenta a origem do valor usado |
| `fonte_tabela` | Tabela do IBGE usada na observação populacional | Permite diferenciar estimativa, Contagem de 2007 e Censo de 2010 |

Quando a população de um município-ano não está publicada, o pipeline usa a última observação anterior disponível. Linhas anteriores à primeira observação populacional do município são excluídas. Essa regra evita preencher um ano passado com informação futura e permite medir quantos valores foram efetivamente observados.

## 8. Divisão temporal

Foi utilizada uma divisão sem embaralhamento:

| Divisão | Regra |
| --- | --- |
| Treinamento | Ano de origem até 2016 |
| Validação | 2017 a 2019 |
| Teste | A partir de 2020 |

Essa escolha simula o uso real: o modelo aprende com o passado e é avaliado em anos posteriores. Uma divisão aleatória poderia colocar observações temporalmente próximas em conjuntos diferentes e gerar uma estimativa otimista por vazamento temporal.

## 9. Modelos avaliados

### Baseline de variação zero

Prevê que a taxa não terá variação no próximo ano. É um ponto de referência importante porque mudanças anuais podem estar concentradas próximas de zero.

### Baseline do último valor

Repete a última variação logarítmica observada (`variacao_log_100_atual`). Não é o mesmo que prever a taxa atual: é um baseline de persistência da última mudança observada.

### Ridge

É uma regressão linear regularizada. Foi incluída por ser relativamente simples de explicar e por servir como referência para relações lineares.

### Random Forest Regressor

É um conjunto de árvores de decisão capaz de representar relações não lineares. A configuração inicial utiliza 150 árvores, folhas mínimas de 10 observações, `random_state=42` e processamento paralelo.

## 10. Métricas

Foram calculadas:

- MAE: erro absoluto médio;
- RMSE: raiz do erro quadrático médio;
- erro mediano absoluto;
- R²;
- MAE percentual e erro mediano percentual quando a taxa-base é positiva.

As métricas principais são avaliadas na escala logarítmica, pois ela é o alvo utilizado no treinamento. As métricas percentuais são complementares e devem ser interpretadas com cautela.

## 11. Resultados da coorte principal

Resultados no conjunto de teste, com 1.357 observações:

| Modelo | MAE log × 100 | RMSE log × 100 | Erro mediano log × 100 | R² | MAE percentual | Mediana percentual |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random Forest | 44,48 | 72,79 | 28,65 | 0,515 | 340,96% | 29,88% |
| Ridge | 58,56 | 87,83 | 38,16 | 0,294 | 357,51% | 36,22% |
| Baseline zero | 70,20 | 112,13 | 33,69 | -0,150 | 370,04% | 31,30% |
| Baseline último valor | 148,53 | 219,42 | 80,26 | -3,404 | 552,31% | 72,48% |

### Interpretação

O Random Forest foi o melhor modelo segundo MAE, RMSE e R² na escala logarítmica. Seu MAE foi aproximadamente 36,6% menor que o baseline de variação zero:

```text
1 − 44,48 / 70,20 ≈ 36,6%
```

O baseline do último valor foi o pior. Isso sugere que a variação de um ano não apresenta persistência simples suficiente para ser repetida no ano seguinte.

O R² de 0,515 indica que, nessa divisão temporal e nessa escala transformada, o Random Forest explica aproximadamente 51,5% da variabilidade do alvo em relação à média do conjunto de teste. Isso não significa 51,5% de probabilidade de acerto e não representa uma probabilidade de ocorrência de homicídios.

O MAE percentual ficou elevado porque alguns casos extremos produzem variações percentuais muito grandes. A mediana percentual, de 29,88%, representa melhor o erro típico do que a média isoladamente.

### Evolução do erro entre os períodos

O MAE logarítmico e o R² dos quatro modelos em cada divisão foram:

| Modelo | Treino MAE | Validação MAE | Teste MAE | Treino R² | Validação R² | Teste R² |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random Forest | 22,43 | 55,15 | 44,48 | 0,492 | -0,165 | 0,515 |
| Ridge | 28,95 | 55,74 | 58,56 | 0,166 | -0,193 | 0,294 |
| Baseline zero | 31,07 | 56,52 | 70,20 | -0,002 | -0,156 | -0,150 |
| Baseline último valor | 51,85 | 73,51 | 148,53 | -1,817 | -0,584 | -3,404 |

O aumento do erro entre treino e os períodos posteriores mostra que o problema muda ao longo do tempo. O Random Forest ainda se recupera no teste em relação à validação, mas isso não deve ser interpretado como prova de estabilidade: são apenas três anos de validação e um recorte de teste iniciado em 2020. A validação em janelas móveis é necessária.

### Erros por porte populacional

No teste da coorte principal, o Random Forest apresentou:

| Porte | Observações | MAE log × 100 | RMSE log × 100 |
| --- | ---: | ---: | ---: |
| 50 mil a 200 mil | 1.047 | 44,67 | 66,69 |
| 200 mil a 500 mil | 212 | 44,49 | 84,30 |
| Acima de 500 mil | 98 | 42,44 | 102,38 |

O MAE é semelhante entre os portes da coorte principal. O RMSE dos municípios acima de 500 mil habitantes é maior que o MAE, indicando a presença de alguns erros extremos. Como esse grupo tem somente 98 observações, suas métricas possuem maior incerteza e não devem ser comparadas sem intervalos de confiança.

### Ganho em relação aos baselines

Na coorte principal, comparando o Random Forest com o baseline de variação zero:

- redução do MAE logarítmico: aproximadamente 36,6%;
- redução do RMSE logarítmico: aproximadamente 35,1%;
- aumento do R²: de -0,150 para 0,515;
- redução do erro mediano percentual: de 31,30% para 29,88%.

O baseline continua importante porque mostra que parte da dificuldade do problema está na própria instabilidade do alvo. O modelo não deve ser considerado útil apenas por produzir previsões; ele precisa superar referências simples em dados futuros.

## 12. Análise de sensibilidade com todos os municípios

Na base completa, o Random Forest apresentou:

- MAE logarítmico: `106,11`;
- RMSE logarítmico: `137,17`;
- R²: `0,464`.

O baseline de variação zero apresentou MAE logarítmico de `123,30`. Portanto, o Random Forest continua superior mesmo quando municípios pequenos são incluídos, mas o erro aumenta consideravelmente.

Essa diferença sustenta a decisão de utilizar a coorte de 50 mil habitantes como análise principal e a base completa como sensibilidade. Ela também mostra que o tamanho populacional está relacionado à estabilidade da taxa e à dificuldade de previsão.

## 13. O que foi possível alcançar

Ao final da Sprint 2, foi possível:

- transformar a base bruta em um painel analítico;
- definir formalmente a unidade de análise e os critérios de qualidade;
- criar um alvo contínuo de previsão de variação;
- evitar comparações temporais inválidas;
- implementar uma divisão temporal realista;
- comparar modelos simples e não lineares;
- identificar o Random Forest como candidato inicial ao MVP;
- quantificar a perda de estabilidade em municípios pequenos;
- gerar previsões e métricas reproduzíveis;
- persistir modelos e versões da execução;
- produzir uma base para o artigo e para o futuro mapa interativo.

## 14. O que ainda não foi alcançado

O trabalho ainda não produz:

- uma probabilidade calibrada de aumento;
- uma classificação definitiva de risco;
- uma explicação causal dos homicídios;
- um modelo pronto para orientar decisões policiais;
- um mapa interativo integrado às previsões;
- intervalos de confiança ou incerteza individual;
- validação espacial e regional suficiente;
- comparação com variáveis socioeconômicas e de segurança pública.

Portanto, o resultado atual é um MVP experimental e acadêmico, não um sistema operacional de alerta.

## 15. Limitações metodológicas

### Taxas pequenas e eventos raros

Em municípios pequenos, poucos eventos podem alterar muito a taxa por 100 mil habitantes. O filtro de 50 mil reduz, mas não elimina, esse problema.

### Denominador populacional

A população integrada é uma variável auxiliar. Não se deve afirmar que ela é exatamente o denominador utilizado pelo Atlas sem obter a informação original da metodologia e dos dados de cálculo.

### Base de variáveis reduzida

As features atuais representam principalmente o histórico da própria taxa e a população. Isso limita a capacidade de explicar fatores econômicos, sociais, demográficos e institucionais associados à violência.

### Período de teste

O teste começa em 2020, período que pode conter alterações atípicas. Um único corte temporal não é suficiente para afirmar estabilidade para qualquer período futuro.

### Métricas percentuais

Quando a taxa atual é muito baixa, a variação percentual bruta pode se tornar enorme. Por isso, MAE percentual e mediana percentual devem ser apresentadas juntas com as métricas logarítmicas.

### Interpretação pública

Uma previsão municipal não determina que um homicídio ocorrerá. Resultados devem ser apresentados com incerteza, contexto e limitações para evitar estigmatização territorial ou uso determinístico.

## 16. Justificativas metodológicas

### Por que não usar apenas “vai aumentar” ou “não vai aumentar”?

Porque o objetivo do projeto é estimar a intensidade da mudança. Uma saída contínua permite ordenar municípios, calcular faixas de variação e posteriormente criar categorias de risco. A classificação binária pode ser adicionada como análise complementar.

### Por que usar a transformação logarítmica?

Porque a porcentagem tradicional não é definida quando a taxa inicial é zero e pode explodir quando a taxa inicial é muito baixa. A diferença de `log1p` mantém essas situações na modelagem e reduz o peso de extremos.

### Por que não fazer uma divisão aleatória?

Porque o objetivo é prever o futuro. A divisão temporal impede que informações de anos posteriores influenciem o treinamento e aproxima a avaliação de um cenário real.

### Por que limitar a população a 50 mil habitantes?

Porque a taxa de municípios pequenos é mais volátil devido ao pequeno número absoluto de eventos. O filtro produz uma análise mais estável, mas reduz a cobertura. Por isso, a base completa foi mantida como análise de sensibilidade.

### O que significa o R² de 0,515?

Não. R² mede a proporção de variabilidade explicada na escala do alvo. Não é acurácia, probabilidade nem percentual de municípios corretamente classificados.

### O modelo pode ser usado diretamente para decisões de segurança pública?

Não neste estágio. O modelo é experimental, não causal e ainda não possui avaliação de incerteza, fairness, calibração ou validação operacional. Ele pode apoiar investigação acadêmica e priorização de análises, mas não deve determinar decisões automaticamente.

### Qual modelo foi escolhido?

O Random Forest é o candidato atual porque apresentou o menor MAE e RMSE e o maior R² na coorte principal. A escolha ainda precisa ser confirmada com validação temporal adicional e análise de robustez.

## 17. Parágrafo-base para o artigo

Foi construído um painel municipal com unidade de análise município-ano, composto por taxas de homicídios por 100 mil habitantes e população municipal. O alvo foi definido como a variação da taxa no ano subsequente, modelada inicialmente pela diferença entre os logaritmos de `1 + taxa` multiplicada por 100. Foram criadas variáveis de defasagem, média e desvio-padrão das três observações históricas anteriores, além da variação histórica mais recente e do logaritmo da população. Para reduzir a instabilidade associada a municípios com poucos eventos, a análise principal considerou observações com população mínima de 50 mil habitantes no ano de origem, mantendo-se a base completa para análise de sensibilidade. O conjunto foi dividido temporalmente em treinamento até 2016, validação de 2017 a 2019 e teste a partir de 2020. Foram comparados dois baselines, uma regressão Ridge e um Random Forest Regressor. O Random Forest apresentou o melhor desempenho no teste da coorte principal, com MAE de 44,48 na escala logarítmica, RMSE de 72,79 e R² de 0,515.

## 18. Próximos passos

1. Avaliar acerto da direção da mudança — aumento ou redução.
2. Criar faixas de risco baseadas na distribuição das previsões.
3. Executar validação temporal em janelas móveis.
4. Avaliar erros por região e estado.
5. Adicionar variáveis socioeconômicas, demográficas e de segurança pública.
6. Testar modelos com contagem de homicídios e taxas estabilizadas.
7. Calcular intervalos de previsão e incerteza.
8. Avaliar importância das variáveis e explicabilidade.
9. Integrar geometrias municipais ao mapa.
10. Definir critérios éticos e de uso responsável antes da publicação do MVP.

## 19. Artefatos e reprodução

Principais arquivos:

- `src/prepare_data.py` — preparação e criação do alvo;
- `src/population.py` — população e filtro de coorte;
- `src/models.py` — modelos e métricas;
- `src/run_mvp_analitico.py` — execução completa;
- `notebooks/sprint2_mvp_analitico.ipynb` — análise documentada;
- `data/processed/model_metrics.csv` — métricas da coorte principal;
- `data/processed/model_metrics_todos.csv` — métricas da base completa;
- `models/model_config.json` — configuração e versões;
- `docs/defesa-sprint2.md` — documento principal com metodologia, resultados e limitações.

Com o ambiente Python configurado, a execução principal é:

```bash
python -m src.run_mvp_analitico
python -m unittest discover -s tests -v
```

## 20. Síntese conclusiva

A Sprint 2 entregou um MVP analítico reproduzível que transforma dados históricos municipais em uma previsão contínua da variação da taxa de homicídios. O Random Forest foi o melhor candidato no recorte de municípios com pelo menos 50 mil habitantes, mas o estudo ainda deve ser entendido como uma investigação preditiva inicial, com limitações de dados, incerteza e generalização que serão tratadas nas próximas sprints.

## 21. Referências e rastreabilidade

- [Atlas da Violência 2024](https://www.ipea.gov.br/atlasviolencia/arquivos/artigos/4600-atlasviolencia2024.pdf)
- [SIDRA/IBGE — Tabela 6579](https://sidra.ibge.gov.br/tabela/6579), população residente estimada
- [SIDRA/IBGE — Tabela 579](https://sidra.ibge.gov.br/tabela/579), Contagem da População de 2007
- [SIDRA/IBGE — Tabela 202](https://sidra.ibge.gov.br/tabela/202), Censo Demográfico
- `data/raw/README.md`, com procedência, data de obtenção e hashes dos arquivos
- `docs/contrato-base-analitica.md`, com o contrato formal da base
