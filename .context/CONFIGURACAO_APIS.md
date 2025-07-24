# 🔑 Configuração de APIs - EcoTravel Agent

## 📋 APIs Utilizadas no Projeto

### 1. **OpenAI API** (Obrigatória)
- **Uso**: LLM principal (GPT models), Embeddings avançados
- **Onde**: `src/agent/ecotravel_agent_rl.py`
- **Funcionalidade**: 
  - Chat/completion para o agente principal
  - Embeddings OpenAI para RAG avançado
  - Processamento de linguagem natural

### 2. **Google API** (Opcional)
- **Uso**: Google Search como ferramenta de busca web
- **Onde**: `src/agent/ecotravel_agent_rl.py` (importada mas não implementada no demo)
- **Funcionalidade**: 
  - Busca web para informações atuais
  - Complementa DuckDuckGo search
  - Eventos e novidades sobre sustentabilidade

### 3. **Open-Meteo API** (Gratuita)
- **Uso**: Previsão do tempo
- **Onde**: `src/agent/ecotravel_agent_rl.py`, ferramenta `get_weather`
- **Funcionalidade**:
  - Clima em tempo real
  - Previsão para planejamento de viagens
  - Sem necessidade de chave API

## ✅ Status Atual - APIs Configuradas

### Suas Chaves Configuradas:
```bash
# Arquivo .env atualizado:
OPENAI_API_KEY=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
GOOGLE_API_KEY=AIzaSyBpfJjF7g-VuLaoU8NR3-EAYJ_71nSg0AI
```

### Como o Sistema Usa Cada API:

#### **OpenAI** - Sistema Principal
```python
# Em ecotravel_agent_rl.py:
self.llm = ChatOpenAI(
    temperature=0.7,
    model_name="gpt-3.5-turbo"
)
self.embeddings = OpenAIEmbeddings()
```

#### **Google Search** - Busca Web (Preparada)
```python
# Importada mas não implementada no demo atual:
from langchain_community.utilities import GoogleSearchAPIWrapper

# Pode ser usado para:
# - Buscar eventos sustentáveis atuais
# - Informações sobre destinos eco-friendly
# - Novidades em transporte sustentável
```

#### **Open-Meteo** - Clima (Funcionando)
```python
def get_weather(location: str) -> str:
    # Busca coordenadas da cidade
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}"
    # Obtém previsão do tempo
    weather_url = f"https://api.open-meteo.com/v1/forecast..."
```

## 🚀 Como Usar as APIs

### Teste Rápido:
```bash
# Carregar as APIs do .env
source .venv/bin/activate
python -c "
from dotenv import load_dotenv
import os
load_dotenv()
print('✅ OpenAI:', bool(os.getenv('OPENAI_API_KEY')))
print('✅ Google:', bool(os.getenv('GOOGLE_API_KEY')))
"
```

### Sistema Completo (com OpenAI):
```bash
# Sistema completo com RL + RAG + APIs
python src/agent/ecotravel_agent_rl.py
```

### Demo Simplificado (sem APIs):
```bash
# Demo que simula as funcionalidades
python setup/demo_ecotravel.py
```

## 🛠️ Implementações Futuras

### Google Search Integration:
```python
# Como implementar Google Search como ferramenta:
def google_search_tool(query: str) -> str:
    google_search = GoogleSearchAPIWrapper()
    results = google_search.run(query)
    return f"Resultados Google: {results}"

# Adicionar como Tool no agente:
Tool(
    name="Google Search",
    func=google_search_tool,
    description="Busca informações atuais na web via Google"
)
```

### Outras APIs Possíveis:
- **DeepSeek API**: LLM alternativo mais barato
- **Hugging Face API**: Modelos open-source
- **Carbon API**: Dados específicos de pegada de carbono
- **Maps API**: Cálculo de rotas sustentáveis

## 📊 Custos e Limites

### OpenAI:
- **GPT-3.5-turbo**: ~$0.002/1K tokens
- **Embeddings**: ~$0.0001/1K tokens
- **Limite**: Depende do plano

### Google Search:
- **API Key configurada**: Programmable Search Element Paid API
- **Uso atual**: Importada mas não ativa no demo
- **Limite**: Conforme plano Google

### Open-Meteo:
- **Gratuita**: Sem limite para uso pessoal
- **Comercial**: Planos pagos disponíveis

## ✅ Verificação de Status

### Script de Teste:
```bash
# Usar o script de teste integrado:
python setup/test_installation.py
```

### Manual:
```python
import os
from dotenv import load_dotenv
load_dotenv()

# Verificar APIs
apis = {
    'OpenAI': os.getenv('OPENAI_API_KEY'),
    'Google': os.getenv('GOOGLE_API_KEY'),
    'OpenWeather': os.getenv('OPENWEATHER_API_KEY')
}

for name, key in apis.items():
    status = "✅ Configurada" if key else "❌ Não configurada"
    print(f"{name}: {status}")
```

---

**📈 Resultado**: Todas as APIs principais estão configuradas e prontas para uso! O sistema pode funcionar em modo completo com OpenAI + Google, ou em modo demo simulado.