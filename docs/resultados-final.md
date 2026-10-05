# Resultados finais e decisões

## 1. Resultado oficial

A implementação final é a versão oficial do MVP. Ela mantém o horizonte de dois
anos e cinco anos de histórico, ajusta hiperparâmetros somente na validação e
amplia a interpretação das previsões no mapa.

| Configuração | Coorte | n | MAE da taxa | MAE percentual | R² log médio |
|---|---|---:|---:|---:|---:|
| Baseline de referência | População ≥ 50 mil | 3.934 | 7,405 | 27,368 p.p. | 0,779 |
| MVP final | População ≥ 50 mil | 3.934 | **7,382** | **27,345 p.p.** | **0,779** |
| Baseline de referência | Todos os municípios | 14.003 | 10,730 | 37,132 p.p. | 0,579 |
| MVP final | Todos os municípios | 14.003 | **10,715** | **37,130 p.p.** | 0,578 |

A melhora em relação ao baseline de referência é pequena e não deve ser apresentada como transformação
radical da capacidade preditiva.

## 2. Decisões metodológicas consolidadas

- a taxa futura média de dois anos é o alvo principal;
- a coorte com população mínima de 50 mil é a referência de estabilidade;
- a coorte ampla é mantida como análise de sensibilidade;
- a validação é temporal e o teste só é consultado na avaliação final;
- a classificação é relativa ao quartil superior do alvo histórico;
- a probabilidade de risco não é probabilidade de homicídio individual;
- faixas e ranking são instrumentos interpretativos, não categorias absolutas;
- o ensemble simples foi descartado como configuração oficial por não apresentar
  ganho material e uniforme;
- o Poisson foi mantido como sensibilidade, pois a contagem é uma proxy derivada
  da taxa;
- a aplicação não é um sistema operacional de segurança pública.

## 3. Contagem Poisson

O Poisson com exposição populacional foi comparado ao baseline histórico:

| Coorte | Modelo | MAE contagem proxy | MAE taxa | MAE percentual |
|---|---|---:|---:|---:|
| População ≥ 50 mil | Baseline histórico | 33,238 | 8,277 | 28,567 p.p. |
| População ≥ 50 mil | Poisson | 37,539 | 9,809 | 47,881 p.p. |
| Todos os municípios | Baseline histórico | 13,126 | 11,534 | 39,137 p.p. |
| Todos os municípios | Poisson | 15,340 | 13,215 | 47,753 p.p. |

O resultado não sustenta usar Poisson como camada principal. Também não sustenta
afirmar que o projeto prevê o número oficial de homicídios.

## 4. Representação no mapa

A aplicação oferece:

- taxa média futura prevista;
- variação percentual prevista;
- probabilidade calibrada de alto risco;
- faixa de risco relativa;
- ranking e percentil relativo;
- largura do intervalo conformal de 90%.

Os intervalos tiveram cobertura retrospectiva aproximada de 86,9% na coorte
principal e 88,6% na coorte ampla. Isso é uma auditoria histórica e não uma
garantia de cobertura futura.

## 5. Previsão futura

A previsão com origem em 2024 produz uma camada separada para 2025–2026. Ela
usa as taxas de 2020–2024 como histórico e os parâmetros escolhidos na
validação temporal. Como ainda não há alvo observado compatível no painel, a
camada é descritiva e não avaliada.

## 6. Comparações preservadas na decisão

O horizonte de quatro anos foi abandonado porque o horizonte de dois anos
apresentou menor erro na coorte principal. O ensemble simples foi avaliado, mas
seu ganho foi marginal e não uniforme. A implementação final incorpora essas
decisões sem manter artefatos históricos concorrendo com o MVP oficial.

## 7. Reprodução e aplicação

```bash
.venv/bin/python -m src.experiment_final
.venv/bin/python -m src.forecast_futura
.venv/bin/python -m unittest discover -s tests -t .
streamlit run app/main.py
```

Para fontes, hashes, regras de disponibilidade e fórmulas, consulte
[`fontes-e-rastreabilidade.md`](fontes-e-rastreabilidade.md). Para a
metodologia completa, consulte [`metodologia-final.md`](metodologia-final.md).
