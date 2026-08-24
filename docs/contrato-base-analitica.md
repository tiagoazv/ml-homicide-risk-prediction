# Contrato da base analítica

## Unidade e chave

Cada linha representa um município em um ano. A chave natural é `municipio_codigo + ano`.

`municipio_codigo` usa o código IBGE com sete dígitos, armazenado como texto para preservar zeros à esquerda. O município não é agregado ou dividido durante o pré-processamento.

## Colunas obrigatórias

| Coluna | Tipo | Unidade/interpretação |
| --- | --- | --- |
| `municipio_codigo` | string | Código IBGE do município |
| `municipio` | string | Nome do município |
| `ano` | inteiro | Ano de referência |
| `taxa_homicidios` | decimal | Taxa de homicídios por 100 mil habitantes |
| `taxa_lag_1` | decimal | Taxa do ano anterior |
| `taxa_lag_2` | decimal | Taxa de dois anos antes |
| `taxa_media_3_anos` | decimal | Média das três taxas históricas |
| `taxa_std_3_anos` | decimal | Desvio-padrão populacional das três taxas históricas |
| `variacao_log_100_atual` | decimal | Variação logarítmica histórica multiplicada por 100 |
| `populacao` | inteiro | População municipal contemporânea ao ano de origem |
| `log_populacao` | decimal | `log1p(populacao)` |
| `porte_populacional` | categoria | Faixas: até 10 mil, 10–50 mil, 50–200 mil, 200–500 mil e acima de 500 mil |
| `target_variacao_log_100` | decimal | Alvo do ano seguinte |
| `target_variacao_percentual` | decimal | Variação percentual bruta quando a taxa atual é positiva |

## Regras de qualidade

- Não pode haver duplicidade na chave município-ano.
- Campos essenciais não podem ser nulos.
- Taxas e populações devem ser não negativas; populações devem ser positivas.
- As janelas históricas e o ano futuro precisam ser consecutivos.
- Linhas anteriores à primeira população disponível para o município são excluídas quando a população é usada.
- Valores populacionais preenchidos pela última observação anterior são identificados por `populacao_observada = False`.
- Nenhuma feature pode utilizar o alvo, o ano futuro ou uma variável publicada depois do período de previsão.
- A coorte principal utiliza `populacao >= 50.000` no ano de origem; a base sem esse filtro é preservada para sensibilidade.

## Critérios de aceite

1. A preparação é executável a partir dos dois arquivos em `data/raw/`.
2. O resultado possui chave município-ano única e tipos compatíveis com este contrato.
3. O pipeline falha explicitamente em duplicidades, valores inválidos ou colunas ausentes.
4. O conjunto de teste é posterior ao treino e à validação.
5. O notebook e os testes reproduzem a preparação e a avaliação sem divisão aleatória.
