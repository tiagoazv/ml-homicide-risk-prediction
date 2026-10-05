# Metodologia final — MVP preditivo

## 1. Escopo

O projeto estima a taxa municipal futura de homicídios e classifica
relativamente os municípios segundo o risco de pertencer ao quartil superior
da taxa futura. A unidade de análise é o município–ano. A saída principal é a
média da taxa nos dois anos seguintes, expressa por 100 mil habitantes.

Este é o protocolo oficial da aplicação. Ele combina ajuste de
hiperparâmetros, avaliação exploratória de Poisson e saídas para interpretação
no mapa. Os artefatos da execução de referência permanecem na pasta técnica
`v11` para manter a rastreabilidade dos resultados já gerados.

## 2. Construção do painel

O painel combina a taxa municipal do Atlas da Violência, população municipal e
identificação territorial do IBGE. O código de seis dígitos do Atlas é associado
ao código municipal IBGE de sete dígitos pelo prefixo territorial. A taxa
publicada não é recalculada.

São exigidos cinco anos históricos consecutivos antes da origem. O horizonte é
de dois anos:

```text
alvo_t = média(taxa_{t+1}, taxa_{t+2})
```

Ausência de uma linha do Atlas é mantida como ausência de observação; ela não é
convertida em zero. A coorte principal exige população igual ou superior a 50
mil habitantes no ano de origem. A coorte ampla inclui todos os municípios
elegíveis.

## 3. Atributos preditivos

O conjunto final possui 17 atributos, derivados somente do histórico disponível
na origem:

- ano, taxa atual, defasagens de um e dois anos;
- média, desvio-padrão, mediana e amplitude da taxa em três e cinco anos;
- tendência linear da taxa e tendência logarítmica;
- variação logarítmica atual;
- número de aumentos no histórico;
- logaritmo da população.

Os indicadores GINI, IDHM, PIND e analfabetismo são disponibilizados na camada
descritiva do mapa. Eles não são apresentados como atributos do modelo final sem
uma nova execução controlada que respeite a disponibilidade temporal de cada
fonte.

## 4. Divisão temporal

Nenhuma divisão usa embaralhamento. As três janelas históricas são:

| Treino | Validação | Teste |
|---|---|---|
| até 2012 | 2013–2014 | 2015–2016 |
| até 2014 | 2015–2016 | 2017–2018 |
| até 2016 | 2017–2018 | 2019–2020 |

Hiperparâmetros, calibração, faixas e intervalos são definidos sem consultar o
teste. O teste é reservado à avaliação final.

## 5. Algoritmos

Foram comparados baselines e modelos explicáveis de regressão:

- baseline da média histórica;
- Ridge, com `alpha=10.0` selecionado;
- Random Forest, com 200 árvores, `min_samples_leaf=20` e
  `max_features="sqrt"`;
- HistGradientBoosting, com `max_iter=100`, `learning_rate=0.1` e
  `max_leaf_nodes=15`.

A escolha é feita por MAE da taxa na validação temporal. O alvo contínuo é
modelado na escala `100 × ln(1 + taxa)`, com transformação inversa para a taxa
apresentada.

A classificação complementar identifica o quartil superior do alvo histórico.
Os classificadores candidatos são selecionados por validação temporal e suas
probabilidades podem ser calibradas. A probabilidade não representa a chance
de ocorrer um homicídio individual.

## 6. Contagem Poisson

O MVP final avalia Poisson com exposição populacional somente como sensibilidade. Como
o Atlas fornece a taxa, e não o numerador oficial compatível com todos os
municípios, a contagem usada é uma proxy:

```text
exposição = população_de_origem × 2 / 100.000
contagem_proxy = taxa_futura_media × exposição
```

O modelo Poisson não superou o baseline da taxa e não é a camada principal da
aplicação. O projeto não afirma prever o número oficial de homicídios.

## 7. Saídas da aplicação

O mapa oferece taxa média futura, variação percentual, probabilidade calibrada
de alto risco, faixa de risco, ranking relativo e largura do intervalo
conformal. A previsão futura de 2024 para 2025–2026 é mantida separada e ainda
não possui avaliação contra valores observados.

## 8. Reprodução

```bash
.venv/bin/python -m src.experiment_final
.venv/bin/python -m src.forecast_futura
.venv/bin/python -m unittest discover -s tests -t .
streamlit run app/main.py
```

Os dados, fórmulas, hashes e limitações estão em
[`fontes-e-rastreabilidade.md`](fontes-e-rastreabilidade.md).
