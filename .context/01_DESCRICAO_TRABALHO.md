# TP 5 - Agentes com LLMs

## Objetivo
Capacitar os alunos a conceber, projetar, implementar e documentar uma solução baseada em agentes autônomos que utilizam Modelos de Linguagem (LLMs). Os alunos devem demonstrar a habilidade de orquestrar um LLM com ferramentas externas e bases de conhecimento para resolver um problema complexo e bem definido.

O projeto deve ser desenvolvido em etapas, conforme descrito abaixo.

## Etapa 1: Proposição e Justificativa do Problema
- **Tarefa**: Escolher um problema do mundo real que possa ser resolvido eficientemente por um sistema de agentes com LLM. O problema deve ser específico e bem definido.
- **Requisitos Fundamentais**:
  - A solução deve obrigatoriamente utilizar **Recuperação de Informação (RAG - Retrieval-Augmented Generation)**. O agente deve consultar uma base de conhecimento (ex.: arquivos em PDF, TXT, CSV, etc.) para responder perguntas ou guiar suas ações.
  - Utilizar pelo menos **uma ferramenta adicional**, como:
    - Acesso a uma API pública (ex.: previsão do tempo, cotação de ações, notícias).
    - Interpretador de código Python para cálculos ou análises.
    - Ferramenta de busca na web (ex.: Google Search, DuckDuckGo).
    - Ferramenta para consultar um banco de dados SQL.
  - **Justificativa**: Descrever claramente o problema e argumentar por que uma arquitetura de agentes com as ferramentas escolhidas é a abordagem mais adequada para resolvê-lo.

## Etapa 2: Desenho da Arquitetura e Fluxo Lógico
- **Tarefa**: Projetar a solução antes da implementação.
- **Fluxograma**:
  - Criar um fluxograma detalhado que ilustre a arquitetura do sistema, contendo:
    - O agente principal (orquestrador).
    - As ferramentas disponíveis (RAG + ferramenta adicional).
    - O fluxo de uma requisição do usuário.
    - Os pontos de decisão onde o agente escolhe qual ferramenta usar (ou se nenhuma é necessária).
    - O fluxo de informação entre o usuário, o agente e as ferramentas.
  - **Descrição Complementar**: Descrever a lógica de funcionamento, incluindo os prompts principais planejados para instruir o agente a raciocinar e tomar decisões.

## Etapa 3: Implementação e Entrega Técnica
- **Ambiente de Execução**:
  - O projeto deve ser entregue como um **notebook executável no Google Colab**.
  - **Alternativa Local**: Caso o Google Colab não seja viável, entregar o projeto em um repositório ou arquivo compactado contendo:
    - Todo o código-fonte.
    - Um arquivo de configuração de ambiente (`requirements.txt` para pip ou `environment.yml` para Conda) que permita recriar o ambiente de execução local.
  - **Boas Práticas**: O código deve ser bem comentado. Evitar expor chaves de API diretamente no código.

## Entregáveis
1. **Documento único (PDF)** contendo:
   - Nomes dos integrantes do grupo.
   - Descrição e justificativa do problema.
   - Fluxograma e descrição da arquitetura.
2. **Notebook (.ipynb)** funcional ou uma pasta compactada com:
   - O projeto local.
   - Arquivo de configuração do ambiente.