# Arquitetura do EcoTravel Agent

## Visão Geral
O sistema utiliza um agente orquestrador baseado no padrão ReAct (Reason + Act) que integra:
- **RAG**: Para consultar guias de viagem sustentáveis e dados de emissões.
- **Ferramentas Externas**: APIs (Open-Meteo), busca web (DuckDuckGo) e interpretador Python.
- **Memória Conversacional**: Mantém contexto entre interações.

## Fluxograma
```plaintext
┌─────────────────────┐
│   Usuário           │
└──────────┬──────────┘
           │ Query
┌──────────▼──────────┐
│  Agente Orquestrador │
│  (ReAct Pattern)     │
└──────────┬──────────┘
           │ Decide: RAG ou Tools
    ┌──────┴──────┬─────────┬──────────┐
    │             │         │          │
┌───▼────┐  ┌────▼────┐ ┌──▼────┐ ┌───▼────┐
│  RAG    │  │ Python  │ │ APIs   │ │ Search │
│ System  │  │ Interp  │ │Externa │ │  Web   │
└─────────┘  └─────────┘ └───────┘ └────────┘
```

## Lógica do Agente
- **Entrada**: Query do usuário (ex.: "Planejar viagem sustentável SP->RJ").
- **Decisão**: O agente usa prompts chain-of-thought para determinar:
  - Se a query requer dados estáticos → RAG (ex.: guias de viagem).
  - Se precisa de cálculos → Python (ex.: pegada de carbono).
  - Se demanda dados em tempo real → API ou busca web (ex.: clima).
- **Execução**: Chama a ferramenta apropriada e sintetiza a resposta.
- **Erro**: Reformula a query ou tenta outra ferramenta se a primeira falhar.

## Prompts Principais
- **Decisão**: "Analise a query '[query]'. Se requer dados de guias ou emissões, use RAG. Se precisa de cálculos, use Python. Se exige clima ou eventos, use API/busca web."
- **RAG**: "Consulte a base de conhecimento e retorne informações relevantes para '[query]' com citação de fontes."
- **Anti-alucinação**: "Verifique a resposta contra as fontes do RAG e corrija discrepâncias."

## Componentes
- **Agente Orquestrador**: LangChain com ReAct, usando Ollama (LLM local).
- **RAG**: LlamaIndex com Sentence-Transformers e FAISS.
- **Ferramentas**:
  - Python: Cálculos de CO2 e otimização de rotas.
  - API Open-Meteo: Previsão do tempo.
  - DuckDuckGo Search: Eventos locais.
- **Memória**: ConversationBufferMemory para contexto.