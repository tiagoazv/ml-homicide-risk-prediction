# Backlog das próximas Sprints

Este arquivo contém issues prontas para serem criadas no GitHub. Os títulos, descrições e checklists seguem o padrão utilizado no backlog atual.

## Sprint 2 — MVP Analítico

Milestone sugerido: Sprint 2 - MVP Analítico

### Criar contrato da base analítica

**Labels:** priority: high, data, ml

Definir o formato final da tabela município-ano que será consumida pelo pipeline e pelos modelos.

- [x] Definir chave município-ano
- [x] Definir colunas obrigatórias
- [x] Documentar tipos e unidades
- [x] Registrar regras para lacunas temporais
- [x] Definir critérios de aceite da base

### Criar pipeline de pré-processamento

**Labels:** priority: high, data, ml

Implementar o pipeline reproduzível de carregamento, padronização e preparação dos dados.

- [x] Carregar dados brutos
- [x] Padronizar nomes e tipos
- [x] Validar duplicidades e valores inválidos
- [x] Registrar transformações aplicadas
- [x] Gerar base analítica processada

### Construir variável-alvo percentual

**Labels:** priority: high, data, ml

Criar a variável contínua que representa a variação da taxa de homicídios no período seguinte.

- [x] Calcular variação percentual entre anos consecutivos
- [x] Tratar taxas-base iguais a zero
- [x] Calcular alternativa logarítmica
- [x] Documentar a escolha principal
- [x] Verificar distribuição e valores extremos

### Criar variáveis históricas preditoras

**Labels:** priority: high, data, ml

Derivar variáveis históricas sem utilizar informações posteriores ao período de previsão.

- [x] Criar taxas defasadas
- [x] Criar médias móveis
- [x] Criar medidas de volatilidade
- [x] Criar variações históricas
- [x] Verificar risco de vazamento temporal

### Integrar população municipal

**Labels:** priority: high, data, research

Avaliar e integrar uma fonte populacional municipal compatível com a taxa e com o período analisado.

- [x] Selecionar fonte oficial
- [x] Padronizar código IBGE
- [x] Verificar cobertura temporal
- [x] Comparar população com a metodologia da taxa
- [x] Documentar limitações da integração

### Definir divisão temporal dos dados

**Labels:** priority: high, ml, research

Definir treinamento, validação e teste respeitando a ordem temporal dos dados.

- [x] Definir período de treinamento
- [x] Definir período de validação
- [x] Definir período de teste
- [x] Evitar divisão aleatória
- [x] Documentar a justificativa metodológica

### Treinar modelo baseline

**Labels:** priority: high, ml

Criar modelos simples de referência para verificar se os modelos de Machine Learning agregam valor.

- [x] Implementar baseline de variação zero
- [x] Implementar baseline baseado no último valor
- [x] Gerar previsões no conjunto de teste
- [x] Registrar métricas
- [x] Documentar limitações

### Treinar modelos de regressão

**Labels:** priority: high, ml

Treinar modelos de regressão compatíveis com a variável-alvo percentual.

- [x] Treinar regressão linear
- [x] Treinar Random Forest Regressor
- [x] Definir parâmetros iniciais
- [x] Registrar versões e configurações
- [x] Salvar modelos de forma reproduzível

### Avaliar e comparar modelos

**Labels:** priority: high, ml, research

Comparar os modelos usando métricas adequadas para a previsão da variação percentual.

- [x] Calcular MAE
- [x] Calcular RMSE
- [x] Calcular erro mediano absoluto
- [x] Comparar contra os baselines
- [x] Analisar erros por tamanho do município
- [x] Selecionar modelo candidato ao MVP

### Documentar a metodologia analítica no artigo

**Labels:** priority: high, article, documentation

Escrever a seção metodológica correspondente ao pipeline e aos experimentos do MVP analítico.

- [x] Descrever fontes e unidade de análise
- [x] Descrever preparação dos dados
- [x] Descrever variável-alvo
- [x] Descrever divisão temporal
- [x] Descrever modelos e métricas
- [x] Registrar decisões e limitações

### Atualizar Kanban e estimativas da Sprint 2

**Labels:** priority: high, project management

Manter o acompanhamento da Sprint 2 atualizado com estimativas e evidências de execução.

- [x] Atualizar status das issues
- [x] Registrar estimativas
- [x] Registrar bloqueios
- [x] Registrar entregas
- [x] Realizar retrospectiva da Sprint

## Sprint 3 — MVP do Produto

Milestone sugerido: Sprint 3 - MVP do Produto

### Definir requisitos funcionais do MVP

**Labels:** priority: high, research, documentation

Definir as funcionalidades mínimas da aplicação que apresentará as previsões.

- [ ] Definir público principal
- [ ] Definir filtros necessários
- [ ] Definir informações exibidas no mapa
- [ ] Definir comportamento para dados ausentes
- [ ] Criar critérios de aceite

### Implementar aplicação Streamlit inicial

**Labels:** priority: high, setup, documentation

Criar a aplicação web inicial para carregar dados e apresentar resultados do modelo.

- [ ] Criar página principal
- [ ] Carregar dados processados
- [ ] Carregar modelo selecionado
- [ ] Exibir instruções de uso
- [ ] Tratar erros de carregamento

### Integrar previsões ao mapa municipal

**Labels:** priority: high, data, setup

Adicionar geometrias municipais e associar cada município às previsões do modelo.

- [ ] Selecionar fonte da geometria
- [ ] Padronizar código IBGE
- [ ] Fazer junção espacial
- [ ] Colorir municípios pela variação prevista
- [ ] Validar municípios sem geometria

### Adicionar filtros e tabela de resultados

**Labels:** priority: high, setup, documentation

Permitir que o usuário explore os resultados por ano, estado, município e faixa de variação.

- [ ] Criar filtro por ano
- [ ] Criar filtro por estado
- [ ] Criar busca por município
- [ ] Exibir tabela ordenável
- [ ] Exibir taxa atual e variação prevista

### Exibir incerteza e limitações da previsão

**Labels:** priority: high, documentation, research

Apresentar a previsão com contexto suficiente para evitar interpretações determinísticas.

- [ ] Exibir definição da variável-alvo
- [ ] Exibir métrica do modelo
- [ ] Informar limitações dos dados
- [ ] Sinalizar taxas-base muito baixas
- [ ] Diferenciar previsão de causalidade

### Criar testes unitários do pipeline

**Labels:** priority: high, setup, ml

Testar as funções de preparação de dados, criação do alvo e geração de previsões.

- [ ] Testar leitura da base
- [ ] Testar padronização de colunas
- [ ] Testar cálculo percentual
- [ ] Testar taxas-base iguais a zero
- [ ] Testar entrada inválida

### Criar testes de integração e aceite

**Labels:** priority: high, setup, documentation

Verificar se o pipeline, modelo e aplicação funcionam juntos.

- [ ] Executar preparação até a previsão
- [ ] Verificar carregamento do modelo
- [ ] Verificar filtros da aplicação
- [ ] Verificar mapa e tabela
- [ ] Definir roteiro de teste de aceite

### Publicar o MVP funcional

**Labels:** priority: high, setup, documentation

Publicar a primeira versão funcional da aplicação para avaliação.

- [ ] Definir plataforma de publicação
- [ ] Configurar execução
- [ ] Configurar arquivos necessários
- [ ] Testar acesso público
- [ ] Registrar limitações da publicação

### Avaliar usabilidade do MVP

**Labels:** priority: medium, research, documentation

Verificar se a aplicação é compreensível e utilizável pelo público definido.

- [ ] Criar roteiro de avaliação
- [ ] Observar tarefas principais
- [ ] Registrar dificuldades
- [ ] Organizar feedback
- [ ] Criar lista inicial de melhorias

### Documentar o desenvolvimento da solução no artigo

**Labels:** priority: high, article, documentation

Descrever a arquitetura, as funcionalidades e as decisões técnicas do MVP.

- [ ] Descrever fluxo de dados
- [ ] Descrever aplicação
- [ ] Descrever mapa e filtros
- [ ] Descrever testes
- [ ] Justificar decisões de design

### Atualizar Kanban e estimativas da Sprint 3

**Labels:** priority: high, project management

Registrar execução, estimativas, bloqueios e resultados da Sprint 3.

- [ ] Atualizar status das issues
- [ ] Registrar esforço realizado
- [ ] Registrar problemas encontrados
- [ ] Registrar versão publicada
- [ ] Realizar retrospectiva da Sprint

## Sprint 4 — Evolução

Milestone sugerido: Sprint 4 - Evolução

### Planejar validação com usuários

**Labels:** priority: high, research, documentation

Definir como o MVP será avaliado sob a perspectiva de usuários e do problema de negócio.

- [ ] Definir perfis de avaliadores
- [ ] Definir hipóteses a testar
- [ ] Criar roteiro de avaliação
- [ ] Definir perguntas de feedback
- [ ] Definir critérios para perseverar ou pivotar

### Coletar feedback do MVP

**Labels:** priority: high, research

Realizar avaliações do MVP e registrar evidências de uso.

- [ ] Aplicar roteiro de avaliação
- [ ] Registrar comentários
- [ ] Registrar dificuldades
- [ ] Registrar sugestões
- [ ] Preservar evidências da coleta

### Analisar feedback e priorizar melhorias

**Labels:** priority: high, research, project management

Transformar o feedback coletado em decisões e tarefas priorizadas.

- [ ] Agrupar problemas semelhantes
- [ ] Classificar impacto e esforço
- [ ] Identificar problemas críticos
- [ ] Definir melhorias da Sprint
- [ ] Registrar decisão de perseverar ou pivotar

### Implementar melhorias do MVP

**Labels:** priority: high, setup, documentation

Implementar as melhorias priorizadas com base nas evidências de validação.

- [ ] Melhorar problemas de usabilidade
- [ ] Corrigir falhas encontradas
- [ ] Melhorar visualizações relevantes
- [ ] Atualizar mensagens e limitações
- [ ] Reexecutar testes

### Reavaliar o modelo após o feedback

**Labels:** priority: high, ml, research

Verificar se o feedback revela problemas de interpretação, desempenho ou utilidade das previsões.

- [ ] Revisar variável-alvo
- [ ] Revisar faixas de variação
- [ ] Analisar erros mais relevantes
- [ ] Verificar necessidade de novas variáveis
- [ ] Registrar mudanças metodológicas

### Revisar Business Model Canvas

**Labels:** priority: high, documentation, research

Atualizar o Canvas com base nos dados, feedback e decisões do projeto.

- [ ] Revisar proposta de valor
- [ ] Revisar segmentos de usuários
- [ ] Revisar canais
- [ ] Revisar recursos e atividades
- [ ] Documentar evidências usadas na revisão

### Escrever resultados, discussão e limitações

**Labels:** priority: high, article, documentation

Consolidar os resultados analíticos e a avaliação do produto no artigo.

- [ ] Apresentar resultados dos modelos
- [ ] Comparar com baselines
- [ ] Discutir erros e limitações
- [ ] Discutir utilidade do MVP
- [ ] Explicitar ausência de inferência causal

### Atualizar Kanban e estimativas da Sprint 4

**Labels:** priority: high, project management

Registrar o ciclo de validação, as melhorias e as decisões tomadas.

- [ ] Atualizar status das issues
- [ ] Registrar feedback e decisões
- [ ] Registrar melhorias implementadas
- [ ] Atualizar riscos
- [ ] Realizar retrospectiva da Sprint

## Sprint 5 — Entrega Final

Milestone sugerido: Sprint 5 - Entrega Final

### Reexecutar o projeto do zero

**Labels:** priority: high, setup, documentation

Verificar se outra pessoa consegue reproduzir os resultados a partir dos arquivos documentados.

- [ ] Criar ambiente limpo
- [ ] Instalar dependências
- [ ] Executar pipeline
- [ ] Executar notebook
- [ ] Reproduzir tabelas e gráficos principais

### Preparar release final

**Labels:** priority: high, setup, documentation

Organizar a versão final do código, dados, modelo e aplicação.

- [ ] Revisar estrutura do repositório
- [ ] Remover arquivos temporários
- [ ] Registrar versão do modelo
- [ ] Revisar documentação de execução
- [ ] Criar release ou tag final

### Documentar reprodutibilidade

**Labels:** priority: high, documentation

Documentar os passos necessários para instalar, executar e compreender o projeto.

- [ ] Documentar instalação
- [ ] Documentar execução do notebook
- [ ] Documentar execução da aplicação
- [ ] Documentar fontes dos dados
- [ ] Documentar limitações e cuidados de interpretação

### Validar coerência entre artigo e projeto

**Labels:** priority: high, article, documentation

Garantir que o artigo descreve exatamente os dados, modelos e resultados entregues.

- [ ] Conferir período dos dados
- [ ] Conferir variável-alvo
- [ ] Conferir modelos e métricas
- [ ] Conferir gráficos e tabelas
- [ ] Atualizar referências

### Preparar apresentação final e pitch

**Labels:** priority: high, documentation

Criar a apresentação final com narrativa clara, evidências e demonstração do produto.

- [ ] Definir problema e motivação
- [ ] Apresentar dados e metodologia
- [ ] Apresentar resultados
- [ ] Demonstrar o mapa
- [ ] Apresentar limitações e próximos passos
- [ ] Ensaiar tempo e divisão da apresentação

### Realizar Sprint Review final

**Labels:** priority: high, project management, documentation

Organizar a revisão final do projeto e registrar o que foi entregue.

- [ ] Demonstrar o MVP final
- [ ] Apresentar evidências das Sprints
- [ ] Comparar objetivo e resultado
- [ ] Registrar pendências conhecidas
- [ ] Coletar avaliação final

### Organizar evidências de participação e gestão

**Labels:** priority: medium, project management, documentation

Reunir as evidências necessárias para demonstrar a gestão do ciclo do projeto.

- [ ] Registrar histórico do Kanban
- [ ] Registrar estimativas
- [ ] Registrar retrospectivas
- [ ] Registrar decisões importantes
- [ ] Organizar contribuições da equipe

### Checklist da entrega final

**Labels:** priority: high, project management, documentation

Verificar todos os critérios da avaliação antes da entrega.

- [ ] Notebook reproduzível
- [ ] Aplicação publicada e funcional
- [ ] Testes executados
- [ ] Artigo alinhado ao projeto
- [ ] README revisado
- [ ] Pitch preparado
- [ ] Release final disponível
