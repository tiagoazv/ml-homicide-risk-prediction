# ML Homicide Risk Prediction

Projeto acadêmico de aprendizado de máquina aplicado à previsão da taxa futura
de homicídios em municípios brasileiros e à classificação relativa de risco.

## Escopo atual

A versão oficial é o **MVP preditivo final** (execução de referência `v11`). O
Atlas da Violência é a fonte principal das taxas municipais. O alvo é a
média da taxa nos dois anos seguintes, expressa por 100 mil habitantes. A
classificação complementar identifica a probabilidade calibrada de o município
pertencer ao quartil superior da taxa futura histórica.

O protocolo usa cinco anos históricos, horizonte de dois anos, validação
temporal e duas coortes: população igual ou superior a 50 mil habitantes e
todos os municípios elegíveis. Ausências de taxa não são convertidas em zero.
O Poisson com exposição populacional permanece somente como análise de
sensibilidade, pois usa uma contagem derivada da taxa e não um numerador oficial.

## Aplicação

A aplicação Streamlit apresenta indicadores observados, previsão retrospectiva
e previsão futura não avaliada de 2024 para 2025–2026. O mapa permite visualizar
taxa futura, probabilidade de alto risco, faixa, ranking, variação percentual e
intervalo de previsão.

```bash
source .venv/bin/activate
streamlit run app/main.py
```

## Estrutura

- `app/` — aplicação e camada de leitura dos artefatos finais;
- `article/` — artigo científico em desenvolvimento;
- `data/raw/` — cópias locais das fontes utilizadas;
- `data/processed/v5/` — indicadores observados para a aplicação;
- `data/processed/v11/` — resultados, métricas e previsões finais;
- `models/v11/` — modelos treinados da versão final;
- `src/` — preparação, modelagem, validação e previsão;
- `tests/` — testes automatizados da base final;
- `docs/` — metodologia, fontes, resultados e documentação da aplicação.

## Documentação consolidada

- [Metodologia final](docs/metodologia-final.md)
- [Fontes e rastreabilidade](docs/fontes-e-rastreabilidade.md)
- [Resultados finais e decisões](docs/resultados-final.md)
- [Previsão futura 2024–2026](docs/previsao-futura.md)
- [Implementação do mapa](docs/plano-mapa.md)
- [Artigo científico](article/artigo.md)
- [Defesa da Sprint 2](docs/defesa-sprint2.md)

## Reprodução

Com os dados brutos locais disponíveis, a sequência principal é:

```bash
.venv/bin/python -m src.experiment_final
.venv/bin/python -m src.forecast_futura
.venv/bin/python -m unittest discover -s tests -t .
```

`experiment_final` gera os artefatos retrospectivos, incluindo a análise de
Poisson e as saídas enriquecidas. `forecast_futura` treina a camada futura
2024–2026, que permanece explicitamente sem avaliação observada.
