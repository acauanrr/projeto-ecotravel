# Roteiro de Implementação

## Fase 1: Preparação (Semana 1)
### Dia 1-2: Setup e Definição
- [ ] Criar ambiente (Colab ou local com Conda).
- [ ] Instalar dependências: `langchain`, `llama-index`, `sentence-transformers`, `faiss-cpu`.
- [ ] Definir escopo exato do problema.
- [ ] Criar estrutura de pastas:
  ```
  projeto/
  ├── notebooks/
  │   ├── 01_exploracao.ipynb
  │   ├── 02_rag_setup.ipynb
  │   └── 03_agent_final.ipynb
  ├── data/
  │   ├── guias/
  │   ├── emissoes/
  │   └── avaliacoes/
  ├── src/
  │   ├── rag/
  │   ├── tools/
  │   └── agent/
  └── docs/
  ```

### Dia 3-4: Coleta de Dados
- [ ] Baixar guias de viagem sustentáveis (ex.: PDFs de Lonely Planet).
- [ ] Criar/coletar CSVs de emissões de CO2.
- [ ] Preparar dados de teste (ex.: JSONs de hotéis).

## Fase 2: Implementação RAG (Semana 2)
### Dia 5-6: RAG Básico
- [ ] Implementar chunking e indexação com LlamaIndex.
- [ ] Testar recuperação básica com queries simples.

### Dia 7-8: Otimizações RAG
- [ ] Adicionar hybrid search (BM25 + Semantic).
- [ ] Implementar reranking (Cohere ou BGE).
- [ ] Configurar estratégias anti-alucinação (Self-Query, fact-checking).

## Fase 3: Integração de Ferramentas (Semana 3)
### Dia 9-10: Ferramentas Externas
- [ ] Configurar Open-Meteo API.
- [ ] Integrar DuckDuckGo Search.
- [ ] Criar calculadora de carbono.

### Dia 11-12: Agente Orquestrador
- [ ] Implementar agente ReAct com LangChain.
- [ ] Configurar prompts chain-of-thought.
- [ ] Testar integração com todas as ferramentas.

## Fase 4: Testes e Validação (Semana 4)
### Dia 13-14: Testes Unitários
```python
test_cases = [
    {
        "input": "Planejar viagem sustentável SP->RJ",
        "expected_tools": ["RAG", "CarbonCalculator", "Weather"],
        "validate": lambda x: "trem" in x.lower()
    }
]
```

### Dia 15-16: Métricas
- [ ] Medir precisão do RAG (Hit Rate, MRR).
- [ ] Avaliar redução de alucinações (comparar com baseline).
- [ ] Calcular tempo de resposta.

## Fase 5: Documentação e Entrega
### Dia 17-18: Documentação
- [ ] Criar fluxograma (Draw.io).
- [ ] Escrever justificativas detalhadas.
- [ ] Preparar PDF final.

### Dia 17-18: Finalização
- [ ] Revisar código e comentários.
- [ ] Criar notebook limpo no Colab.
- [ ] Preparar apresentação.

## Código Exemplo
```python
# main_agent.py
from langchain.agents import initialize_agent, AgentType
from langchain.llms import Ollama
from langchain.memory import ConversationBufferMemory

class EcoTravelAgent:
    def __init__(self):
        self.llm = Ollama(model="llama3")
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.tools = [
            # RAG tool,
            # carbon_tool,
            # weather_tool,
            # search_tool
        ]
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT,
            memory=self.memory,
            verbose=True
        )
    
    def run(self, query):
        return self.agent.run(query)

# Uso
agent = EcoTravelAgent()
result = agent.run("Quero viajar de São Paulo para o Rio de forma sustentável")
```