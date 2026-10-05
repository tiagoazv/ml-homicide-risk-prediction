# Fontes e rastreabilidade

## 1. Fontes utilizadas

| Arquivo local | Fonte e uso |
|---|---|
| `data/raw/Taxa de homicídios registrados_municípios.csv` | Exportação local fornecida pelo autor a partir do [Atlas da Violência](https://www.ipea.gov.br/atlasviolencia/tema/1/); taxa municipal de 2000–2024 e alvo do modelo. |
| `data/raw/ADH_GINI.csv` | Atlas do Desenvolvimento Humano/Ipeadata; indicador GINI para a camada descritiva. |
| `data/raw/ADH_IDHM.csv` | Atlas do Desenvolvimento Humano/Ipeadata; indicador IDHM para a camada descritiva. |
| `data/raw/ADH_PIND.csv` | Atlas do Desenvolvimento Humano/Ipeadata; extrema pobreza para a camada descritiva. |
| `data/raw/ADH_T_ANALF15M.csv` | Ipeadata/IBGE; analfabetismo da população de 15 anos ou mais para a camada descritiva. |
| `data/raw/populacao_municipal_ibge_2001_2021.csv` | IBGE/SIDRA, principalmente tabela 6579, com complementos das tabelas 579 e 202; população histórica, filtro e exposição exploratória. |
| `data/raw/sidra_populacao_2022.json` | IBGE/SIDRA, tabela 4714; população de 2022. |
| `data/raw/sidra_populacao_2024.json` | IBGE/SIDRA, tabela 6579; estimativa populacional de 2024 para a previsão futura. |
| `data/raw/municipios_ibge.json` | Cadastro de localidades do [IBGE](https://servicodados.ibge.gov.br/api/v1/localidades/municipios); nomes e códigos municipais. |
| `data/geo/municipios_2022.geojson` | Malha municipal do IBGE simplificada para visualização no mapa. |
| `data/raw/ipeadata_metadados.json` | Metadados consultados para confirmar conceitos, unidades e referências temporais dos indicadores sociais. |

O arquivo do Atlas é uma cópia local fornecida pelo autor. A URL do tema está
registrada, mas a data e o endereço exato da exportação não foram preservados.
Portanto, o projeto cita o portal oficial como fonte conceitual, mas não afirma
reprodução byte a byte do download original.

## 2. Regras de disponibilidade

- uma ausência no Atlas não é convertida em zero;
- indicadores censitários mantêm o ano da referência original;
- a referência temporal de um indicador não é confundida com sua data de
  publicação;
- dados posteriores ao ano de origem não entram nas variáveis preditivas;
- a origem futura 2024 exige as taxas consecutivas de 2020 a 2024;
- 2025 e 2026 não são usados como rótulos observados na previsão futura.

Os indicadores sociais foram integrados somente para visualização descritiva.
Uma futura inclusão deles no modelo exigirá declarar, para cada período, a data
de disponibilidade da versão da fonte e repetir a validação temporal.

## 3. Fórmulas principais

### Taxa futura

```text
taxa_futura_media_2_anos = (taxa_{t+1} + taxa_{t+2}) / 2
```

### Transformação do alvo

```text
alvo_modelo = 100 × ln(1 + taxa_futura_media_2_anos)
taxa_prevista = exp(previsao_modelo / 100) − 1
```

### Contagem Poisson exploratória

```text
exposição_2_anos = população_de_origem × 2 / 100.000
contagem_proxy_2_anos = taxa_futura_media_2_anos × exposição_2_anos
```

A contagem acima é derivada da taxa e não é numerador oficial do Atlas, do SIM
ou do DATASUS.

## 4. Artefatos finais

- `data/processed/v5/indicadores_mapa.csv`: indicadores observados e referências
  temporais usados pela camada descritiva;
- `data/processed/v11/painel.csv`: painel município–ano com atributos e alvo;
- `data/processed/v11/tuning_taxa_validacao.csv`: candidatos avaliados somente
  na validação;
- `data/processed/v11/metricas_taxa_suavizada_janelas.csv`: desempenho por
  modelo, coorte e janela;
- `data/processed/v11/metricas_contagem_v11.csv`: análise Poisson proxy;
- `data/processed/v11/previsoes_mapa.csv`: camada retrospectiva usada pela
  aplicação;
- `data/processed/v11/previsoes_futuras_2024_2026.csv`: camada futura não
  avaliada;
- `data/processed/v11/config.json` e `config_futuro.json`: parâmetros, fontes,
  tamanhos e hashes registrados na execução.

## 5. Limitações

As taxas do Atlas não devem ser tratadas como uma série com cobertura completa
sem documentação adicional do produtor. População e taxa podem ter definições
e calendários de disponibilidade diferentes. As previsões são retrospectivas
no período de teste e a camada 2024–2026 só poderá ser avaliada quando houver
dados observados correspondentes.

As fontes conceituais incluem o [Atlas da Violência](https://www.ipea.gov.br/atlasviolencia/tema/1/),
o [SIDRA/IBGE — tabela 6579](https://sidra.ibge.gov.br/tabela/6579), a
[tabela SIDRA 4714](https://sidra.ibge.gov.br/tabela/4714) e os metadados do
[Ipeadata](https://www.ipeadata.gov.br/api/odata4/Metadados).
