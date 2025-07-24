# Arquitetura do EcoTravel Agent

## Visão Geral

O EcoTravel Agent é um sistema complexo de IA que integra múltiplas tecnologias para fornecer recomendações de viagens sustentáveis. A arquitetura é baseada no padrão de agentes inteligentes com capacidades RAG (Retrieval-Augmented Generation).

## Diagrama de Arquitetura

```
┌─────────────────────────────────────────────────────────────┐
│                    USUÁRIO                                  │
│              (Interface de Chat)                            │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                AGENTE ORQUESTRADOR                          │
│              (Padrão ReAct + LLM)                           │
│  • Processamento de linguagem natural                       │
│  • Planejamento de ações                                    │
│  • Coordenação de ferramentas                               │
└─────────────────────┬───────────────────────────────────────┘
                      │
      ┌───────────────┼───────────────┐
      │               │               │
┌─────▼─────┐  ┌─────▼─────┐  ┌─────▼─────┐  ┌─────────────┐
│  SISTEMA  │  │FERRAMENTAS│  │   APIs    │  │   MEMÓRIA   │
│    RAG    │  │CUSTOMIZADAS│  │ EXTERNAS  │  │CONVERSAÇÃO │
└─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └─────────────┘
      │              │              │
┌─────▼─────┐  ┌─────▼─────┐  ┌─────▼─────┐
│Base Conhe-│  │• Carbon   │  │• Clima    │
│cimento    │  │  Calc     │  │  (OpenM.) │
│• Guias    │  │• Python   │  │• Busca    │
│• Emissões │  │  Interp   │  │  Web      │
│• Hotéis   │  │• Análise  │  │  (DDG)    │
└───────────┘  └───────────┘  └───────────┘
```

## Componentes Principais

### 1. Agente Orquestrador (`src/agent/eco_travel_agent.py`)

**Responsabilidades:**
- Interpretar consultas do usuário
- Planejar sequência de ações
- Coordenar uso de ferramentas
- Sintetizar respostas finais
- Gerenciar contexto conversacional

**Tecnologias:**
- LangChain ReAct Agent
- LLM (Ollama/OpenAI/HuggingFace)
- Memória conversacional

**Fluxo de Execução:**
1. **Análise**: Interpreta a consulta do usuário
2. **Planejamento**: Decide quais ferramentas usar
3. **Execução**: Chama ferramentas em sequência
4. **Síntese**: Combina resultados em resposta coerente

### 2. Sistema RAG (`src/rag/rag_system.py`)

**Função:** Recuperação de informações da base de conhecimento

**Estratégias Implementadas:**
- **Chunking Semântico**: Divisão inteligente de documentos
- **Busca Híbrida**: Combinação de BM25 (lexical) + embeddings (semântica)
- **Reranking**: Re-ordenação de resultados para maior precisão
- **Anti-alucinação**: Citação de fontes e verificação

**Pipeline RAG:**
```
Documentos → Chunking → Embeddings → Índices (FAISS + BM25)
                                         ↓
Query → Busca Híbrida → Reranking → Contexto para LLM
```

**Configurações:**
- Chunk size: 512 caracteres
- Overlap: 50 caracteres
- Embedding model: sentence-transformers/all-MiniLM-L6-v2
- Top-K: 10 resultados

### 3. Ferramentas Customizadas

#### 3.1 Carbon Calculator (`src/tools/carbon_calculator.py`)

**Função:** Cálculo preciso de pegadas de carbono

**Características:**
- Fatores de emissão por modal de transporte
- Correções por tipo de viagem e ocupação
- Comparação entre modais
- Recomendações sustentáveis
- Cálculo de compensação

**Dados:**
```python
emissions = {
    "aviao": 0.255,    # kg CO2/km
    "carro": 0.171,
    "onibus": 0.089, 
    "trem": 0.041,
    "bicicleta": 0.0
}
```

#### 3.2 Weather API (`src/tools/weather_api.py`)

**Função:** Informações meteorológicas atualizadas

**Recursos:**
- Clima atual e previsões
- Análise de impacto na viagem
- Recomendações baseadas no clima
- Integração com Open-Meteo API

#### 3.3 Web Search (`src/tools/web_search.py`)

**Função:** Informações atualizadas da web

**Capacidades:**
- Busca geral com DuckDuckGo
- Busca de informações de viagem
- Eventos locais
- Opções sustentáveis

### 4. Base de Conhecimento (`data/`)

**Estrutura:**
```
data/
├── guias/                 # Guias de viagem sustentável
│   └── guia_sustentavel_brasil.txt
├── emissoes/             # Dados de emissões de transporte
│   └── emissoes_transporte.csv
└── avaliacoes/           # Avaliações de hotéis sustentáveis
    └── hoteis_sustentaveis.json
```

**Conteúdo:**
- Guias de viagem sustentável para o Brasil
- Tabelas de emissões por modal de transporte
- Base de hotéis sustentáveis com certificações
- Dicas de atividades ecológicas

## Padrões e Estratégias

### 1. Padrão ReAct (Reasoning + Acting)

O agente segue o padrão ReAct para tomada de decisões:

```
Thought: [Raciocínio sobre o que fazer]
Action: [Ferramenta a ser usada]
Action Input: [Parâmetros da ferramenta]
Observation: [Resultado da ferramenta]
... (repete até ter informação suficiente)
Final Answer: [Resposta final ao usuário]
```

### 2. Estratégia RAG Avançada

**Busca Híbrida:**
- **BM25 (Lexical)**: Correspondência exata de termos
- **Semantic Search**: Similaridade semântica via embeddings
- **Combinação**: Score final = α × semantic + (1-α) × lexical

**Reranking:**
- Cross-encoder para re-ordenação
- Consideração do contexto da query
- Melhoria da precisão final

### 3. Prompts Especializados

**System Prompt:**
```
Você é o EcoTravel Agent, especializado em viagens sustentáveis.

PROCESSO:
1. ANÁLISE: Entenda a solicitação
2. PESQUISA: Use ferramentas apropriadas  
3. CÁLCULO: Calcule pegadas de carbono
4. COMPARAÇÃO: Compare opções
5. RECOMENDAÇÃO: Priorize sustentabilidade
6. CONTEXTO: Inclua clima e eventos

DIRETRIZES:
- SEMPRE priorize sustentabilidade
- Calcule e compare emissões
- Seja específico com números
- Explique o "porquê" das recomendações
```

## Performance e Escalabilidade

### Métricas de Performance

**Sistema RAG:**
- Tempo de construção de índice: ~5-10s
- Tempo de busca: ~0.1-0.2s
- Precisão (Hit Rate): >80%

**Ferramentas:**
- Cálculo de carbono: <10ms
- API de clima: ~1-2s
- Busca web: ~2-3s

**Agente Completo:**
- Resposta típica: 5-15s
- Uso de memória: ~500MB-1GB

### Otimizações Implementadas

1. **Caching de Embeddings**: Evita recálculo
2. **Índices Otimizados**: FAISS para busca rápida
3. **Lazy Loading**: Carregamento sob demanda
4. **Paralelização**: Múltiplas consultas simultâneas

## Extensibilidade

### Adicionando Novas Ferramentas

```python
from langchain.tools import Tool

def new_tool_function(input_str: str) -> str:
    # Implementação da ferramenta
    return result

new_tool = Tool(
    name="NewTool",
    func=new_tool_function,
    description="Descrição da ferramenta"
)

# Adicionar à lista de ferramentas do agente
agent.tools.append(new_tool)
```

### Expandindo Base de Conhecimento

1. Adicionar novos documentos em `data/`
2. Executar `rag.build_index()` para reprocessar
3. Novos dados serão automaticamente incluídos

### Customizando Modelos

```python
# Usar modelo local diferente
agent = EcoTravelAgent(model_name="ollama:llama2")

# Configurar embedding customizado
rag = AdvancedRAGSystem(
    embedding_model="custom-model-name"
)
```

## Segurança e Privacidade

### Dados do Usuário
- Não armazenamento permanente de consultas
- Memória conversacional limitada (10 turnos)
- Não rastreamento de localização

### APIs Externas
- Uso de APIs gratuitas quando possível
- Fallbacks para funcionalidade offline
- Rate limiting respeitado

### Validação de Entrada
- Sanitização de inputs do usuário
- Validação de parâmetros de ferramentas
- Tratamento de erros robusto

## Dependências e Requisitos

### Dependências Core
```
langchain>=0.1.0
sentence-transformers>=2.2.0
faiss-cpu>=1.7.4
pandas>=2.1.0
numpy>=1.24.0
```

### Dependências Opcionais
```
ollama              # Para LLM local
openai>=1.6.0       # Para GPT
duckduckgo-search   # Para busca web
gradio>=4.0.0       # Para interface web
```

### Requisitos de Sistema
- Python 3.8+
- RAM: 4GB mínimo, 8GB recomendado
- Armazenamento: 2GB para modelos
- Internet: Para APIs externas

## Monitoramento e Debugging

### Logs de Execução
- Cada passo do agente é logado
- Tempos de execução rastreados
- Erros capturados e reportados

### Métricas de Qualidade
- Hit rate do sistema RAG
- Tempo de resposta por componente
- Taxa de sucesso das ferramentas

### Debugging Tools
```python
# Execução verbosa
agent = EcoTravelAgent(verbose=True)

# Análise de passos
result = agent.run("query")
print(result['execution_steps'])

# Benchmark do sistema
python src/tests/test_system.py --benchmark
```

Esta arquitetura permite alta flexibilidade, performance otimizada e fácil extensibilidade, mantendo o foco na qualidade das recomendações sustentáveis.