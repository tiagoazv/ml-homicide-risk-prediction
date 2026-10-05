# Implementação da aplicação de indicadores municipais

## Primeira entrega

Aplicação Streamlit com mapa municipal, filtros de ano, UF, município e
indicador, ficha municipal, exportação da tabela filtrada, camada retrospectiva
de teste e camada futura não avaliada. A camada descritiva usa os artefatos v5;
a camada preditiva usa os artefatos da execução final e identifica origem, horizonte, coorte e
divisão de teste.

O plano aproveita Streamlit, já previsto no projeto, e a integração documentada com PyDeck. Geometrias municipais simplificadas do IBGE serão baixadas uma vez e mantidas localmente, com edição e hash registrados. A implementação precisa conferir a versão instalada antes de usar APIs de seleção de polígonos. Nenhuma dependência foi atualizada nesta etapa.

## Atualização da interface do MVP final

A aplicação atual passou a oferecer controles adicionais para a leitura do
mapa:

- filtro por faixa de risco, percentil de ranking e população mínima;
- alternância entre camada automática, pontos e polígonos;
- escala amarelo claro → laranja → vermelho escuro para taxas e métricas
  preditivas sequenciais; a variação percentual mantém escala divergente;
- escala de cores fixa ou robusta, calculada entre os percentis 1 e 99 dos
  valores visíveis;
- viewport ajustada automaticamente ao recorte atual, inclusive por UF;
- mapa-base Carto claro para fornecer contexto geográfico.
- série temporal municipal com eixo fixo de 2000 a 2030; anos sem observação
  permanecem sem ponto e não são preenchidos por interpolação.
- gráfico temporal híbrido: histórico observado em cinza, média prevista em
  laranja tracejado e intervalo conformal de 90% em faixa translúcida; na
  camada retrospectiva, um losango azul marca a média observada do horizonte.

O tooltip de cada município identifica a métrica exibida, sua unidade e uma
interpretação curta. Assim, “Valor” não é tratado como uma medida universal:
nos indicadores observados, pode ser taxa por 100 mil habitantes, população,
índice ou percentual; nas previsões, pode ser taxa futura, variação percentual,
probabilidade de alto risco, percentil de ranking ou largura do intervalo de
previsão. A referência temporal e a situação do dado continuam apresentadas
separadamente.

A escala robusta altera somente a representação cromática. Os valores da
tabela, do tooltip e do download permanecem inalterados. O ranking continua
sendo calculado na coorte original; os filtros apenas selecionam quais linhas
serão exibidas.

No gráfico temporal, a linha laranja representa a média prevista para o
horizonte de dois anos, não duas previsões anuais independentes. A faixa e as
linhas tracejadas delimitam o intervalo de incerteza conformal de 90% dessa
média. Na camada futura não há losango azul, porque ainda não existe valor
observado para comparação.

## Dados já entregues

| Arquivo em `data/processed/v5/` | Uso |
| --- | --- |
| `indicadores_mapa.csv` | Grade descritiva de 2000–2024; valores, referências, idade e ausência de taxa |
| `config.json` e `auditoria_geometria.json` | Manifestos e verificações da camada descritiva |
| `data/processed/v11/previsoes_mapa.csv` | Somente teste, com regressão e classificação selecionadas na validação |
| `data/processed/v11/metricas_taxa_suavizada_janelas.csv` | Métricas da taxa média dos dois anos futuros |
| `data/processed/v11/metricas_risco_alto_janelas.csv` | Métricas da classificação de alto risco |
| `data/processed/v11/selecao_*_validacao.csv` | Identificação reproduzível dos candidatos selecionados |
| `data/processed/v11/previsoes_futuras_2024_2026.csv` | Projeção futura não avaliada |

As geometrias estão em `data/geo/municipios_2022.geojson`, com auditoria em `auditoria_geometria.json`. A aplicação mantém essa malha original como fonte e cria, em memória, uma cópia simplificada exclusivamente para o desenho em escala nacional. A aplicação está implementada localmente em `app/`; publicação externa ainda depende de revisão.

Para reduzir a carga do WebGL, a aplicação usa pontos municipais coloridos na
visão nacional por padrão e carrega os polígonos simplificados quando uma UF é
selecionada. A camada também pode ser alternada manualmente entre pontos e
polígonos. O mapa-base Carto claro fornece referência espacial, mas a camada
temática continua funcionando como dado local. Os municípios sem valor
continuam na tabela, nos contadores e no download; essa otimização visual não
altera os dados analíticos.

## Regras de apresentação

- Manter todos os municípios da malha escolhida em uma junção à esquerda por código IBGE de sete dígitos. Município sem registro deve continuar disponível na tabela, no download e nos contadores, com motivo da ausência; para reduzir a carga do WebGL, pode ser omitido da camada visual.
- A camada de homicídios apresenta a taxa publicada, por 100 mil habitantes. Nunca converter ausência em zero.
- Cada indicador social mostra unidade, fonte, ano de referência e idade. Por exemplo: “GINI 0,52 — referência 2010”, mesmo quando o filtro estiver em 2024.
- A população reúne a série 2001–2021 e as tabelas SIDRA de 2022/2024. Não há população anual observada de 2023; valores de referência carregados devem mostrar o ano de disponibilidade.
- Disponibilizar modo “somente observações do ano” e modo “última referência até o ano”. No primeiro, omitir o valor retido, mantendo a ausência explícita na tabela e nos contadores.
- A camada retrospectiva usa somente `data/processed/v11/previsoes_mapa.csv`. A camada futura usa `previsoes_futuras_2024_2026.csv`; ambas mostram origem e horizonte e não são apresentadas como alertas operacionais.
- Selecionar coorte principal ou ampla, com contagem de municípios cobertos. A ampla também exige histórico e indicadores válidos; não cobre o país inteiro.
- Exibir taxa média futura prevista, variação percentual, probabilidade calibrada de alto risco e desempenho no teste. Não rotular a porcentagem prevista como probabilidade.
- Não apresentar intervalos individuais que ainda não foram calculados. Erro médio do conjunto não é intervalo de previsão municipal.
- Usar escala amarelo claro → vermelho escuro para a taxa de homicídios e as
  métricas preditivas em que valores maiores indicam maior intensidade; manter
  escala divergente centrada em zero para variação. Indicadores sociais que não
  têm interpretação monotônica de risco mantêm escala neutra. Fixar limites por
  camada para que trocar o ano não mude arbitrariamente a interpretação das
  cores.
- Oferecer tabela acessível junto ao mapa e informar cobertura e ausências para os filtros ativos.

## Backlog priorizado

### MAP-01 — Preparar e auditar geometrias (P0, 3 pontos)

- [ ] Selecionar e registrar edição da malha do IBGE e licença/atribuição.
- [ ] Baixar GeoJSON simplificado e armazenar em `data/geo/`.
- [ ] Validar códigos, CRS e polígonos; gerar relatório de códigos sem dados ou sem geometria.
- [ ] Documentar limitações ao usar uma única malha em séries históricas, incluindo criação/desmembramento de municípios.

Aceite: nenhuma linha é associada por nome; códigos sem correspondência aparecem no relatório e municípios sem dados continuam na malha.

### MAP-02 — Construir tela descritiva (P0, 5 pontos; depende de MAP-01)

- [ ] Criar `app/main.py` e funções de carregamento/cache em `app/data.py`.
- [ ] Implementar filtros e mapa PyDeck com polígonos, legenda e tooltip.
- [ ] Exibir indicadores, fontes, referências e dados ausentes.
- [ ] Acrescentar ficha municipal, séries temporais e download CSV.

Aceite: alternar GINI 2010, analfabetismo 2022 e taxa 2024 altera valores e rótulos corretamente; município sem taxa continua selecionável. Não interpolar censos na série temporal.

### MAP-03 — Integrar previsão retrospectiva (P0, 3 pontos; depende de MAP-02) — concluído

- [x] Carregar apenas os candidatos selecionados por validação.
- [x] Implementar filtros de coorte, origem e horizonte futuro.
- [x] Exibir taxa prevista, variação, probabilidade de alto risco e resultado observado no alvo como avaliação retrospectiva.
- [x] Mostrar métricas agregadas e a ausência de intervalo individual.

Aceite: nenhum ponto de treino/validação é exibido como teste; nenhum município recebe previsão inexistente; a cobertura efetiva varia por janela e está registrada nos arquivos de métricas da execução final.

### MAP-04 — Verificar e disponibilizar demonstração (P1, 3 pontos)

- [ ] Testar junções, filtros, tratamento de ausências e referências temporais.
- [ ] Avaliar legibilidade e uso em tela pequena com usuários da disciplina.
- [ ] Medir tempo de carregamento e tamanho da malha; cachear geometrias sem duplicá-las por ano.
- [ ] Fixar dependências da aplicação após testar compatibilidade.
- [ ] Preparar instruções `streamlit run app/main.py` e decidir hospedagem.

Aceite: demonstração local repetível, com artefatos identificados; publicação somente depois da revisão do produto. Os 14 pontos são estimativa relativa, não prazo garantido.

## Fluxo previsto

```text
CSV oficiais → pipeline analítico → camada descritiva e camada preditiva final
Malha IBGE original → geometria simplificada para desenho → junção por código
Camadas descritiva/preditiva + geometrias → filtros → mapa + ficha + exportação
```

Para uma camada de previsão operacional ainda será necessário produzir novos dados de entrada, sem exigir um alvo já conhecido, e avaliar a estabilidade fora da amostra retrospectiva. A camada entregue é de avaliação histórica, distinta de um alerta futuro.

## Documentação técnica consultada

- [Streamlit: st.pydeck_chart](https://docs.streamlit.io/develop/api-reference/charts/st.pydeck_chart).
- [IBGE: API de malhas geográficas](https://servicodados.ibge.gov.br/api/docs/malhas?versao=3).
