# Acompanhamento e retrospectiva — Sprint 2

## Estimativa

- Estimativa inicial: 8 pontos de esforço para o núcleo analítico.
- Escopo adicional executado: 5 pontos para integração populacional, persistência de modelos, análise por porte e documentação.

## Entregas

- Contrato da base analítica.
- Pipeline de preparação com população municipal.
- Dois baselines, Ridge e Random Forest.
- Métricas gerais e por porte populacional.
- Modelos serializados e configuração de execução.
- Testes automatizados e notebook executado ponta a ponta.
- Metodologia inicial para o artigo.

## Bloqueios e decisões

- A API do SIDRA recusou uma consulta de múltiplos períodos; a série foi extraída ano a ano.
- A Tabela 6579 não possui observações para todos os códigos em todos os anos. O pipeline usa retenção da última observação anterior e marca essa origem.
- O ano de 2007 usa a Contagem da População e 2010 usa o Censo, com as fontes registradas no arquivo bruto.

## Retrospectiva

- Funcionou: separação temporal, validações explícitas e artefatos reproduzíveis.
- Melhorar: fixar versões em um arquivo de ambiente e investigar a diferença entre o denominador do Atlas e a população integrada.
- Próxima ação: enriquecer o conjunto com indicadores socioeconômicos e iniciar o MVP visual da Sprint 3.
