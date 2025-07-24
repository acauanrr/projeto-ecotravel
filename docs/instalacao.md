# Guia de Instalação - EcoTravel Agent

## Pré-requisitos

### Sistema Operacional
- Windows 10/11, macOS 10.15+, ou Linux (Ubuntu 18.04+)
- Python 3.8 ou superior
- Git (para clonar o repositório)

### Hardware Recomendado
- **RAM**: 8GB mínimo (4GB pode funcionar com limitações)
- **Armazenamento**: 5GB livres (2GB para modelos + 3GB para dependências)
- **Processador**: CPU multi-core recomendado para melhor performance
- **Internet**: Conexão estável para downloads e APIs

## Instalação Rápida

### 1. Clone o Repositório

```bash
git clone https://github.com/seu-usuario/ecotravel-agent.git
cd ecotravel-agent
```

### 2. Crie um Ambiente Virtual

```bash
# Python venv
python -m venv venv

# Ativar no Windows
venv\Scripts\activate

# Ativar no macOS/Linux
source venv/bin/activate
```

### 3. Instale as Dependências

```bash
# Dependências principais
pip install -r requirements.txt

# Dependências opcionais (recomendado)
pip install gradio streamlit duckduckgo-search
```

### 4. Configure as Variáveis de Ambiente

```bash
# Copie o arquivo de exemplo
cp .env.example .env

# Edite conforme necessário
nano .env  # ou seu editor preferido
```

### 5. Teste a Instalação

```bash
# Teste básico
python -c "from src.agent.eco_travel_agent import EcoTravelAgent; print('✅ Instalação OK!')"

# Teste completo
python src/tests/test_system.py --tests
```

## Instalação Detalhada

### Opção 1: Ambiente Conda (Recomendado)

```bash
# Criar ambiente conda
conda create -n ecotravel python=3.9
conda activate ecotravel

# Instalar dependências científicas via conda
conda install numpy pandas scikit-learn matplotlib seaborn jupyter

# Instalar dependências específicas via pip
pip install -r requirements.txt
```

### Opção 2: Instalação com Poetry

```bash
# Instalar Poetry (se não tiver)
curl -sSL https://install.python-poetry.org | python3 -

# Instalar dependências
poetry install

# Ativar ambiente
poetry shell
```

### Opção 3: Docker (Para Produção)

```bash
# Construir imagem
docker build -t ecotravel-agent .

# Executar container
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  -p 8000:8000 \
  ecotravel-agent
```

## Configuração de LLM

O EcoTravel Agent suporta múltiplos modelos de linguagem. Configure pelo menos uma das opções:

### Opção 1: Ollama (Local, Gratuito)

```bash
# Instalar Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Baixar modelo (escolha um)
ollama pull llama3        # Recomendado: rápido e eficiente
ollama pull llama2        # Alternativa menor
ollama pull mistral       # Boa para idiomas

# Testar
ollama run llama3 "Olá, você está funcionando?"
```

**Configuração no .env:**
```bash
LOCAL_MODEL_NAME=llama3
LOCAL_MODEL_HOST=localhost:11434
```

### Opção 2: OpenAI API (Pago, Mais Poderoso)

```bash
# Obter API key em https://platform.openai.com/api-keys
```

**Configuração no .env:**
```bash
OPENAI_API_KEY=sk-sua-chave-aqui
```

### Opção 3: Modelo Local com HuggingFace (Gratuito, Mais Lento)

```bash
# Instalar transformers
pip install transformers torch torchvision torchaudio
```

**O sistema detectará automaticamente e usará modelos disponíveis.**

## Configuração de APIs Externas

### API de Clima (Open-Meteo - Gratuito)

**Não requer configuração** - funciona automaticamente.

Opcionalmente, para WeatherAPI (mais features):
```bash
# Obter chave gratuita em https://www.weatherapi.com/
```

**No .env:**
```bash
OPENWEATHER_API_KEY=sua-chave-aqui
```

### Busca Web (DuckDuckGo - Gratuito)

```bash
# Instalar dependência
pip install duckduckgo-search
```

**Nenhuma configuração adicional necessária.**

## Verificação da Instalação

### Teste Rápido

```python
# Execute no Python
from src.agent.eco_travel_agent import EcoTravelAgent

# Inicializar agente
agent = EcoTravelAgent()

# Teste simples
result = agent.run("Qual a emissão de CO2 de um voo de 400km?")
print(result['response'])
```

### Teste Completo

```bash
# Executar todos os testes
python src/tests/test_system.py --report

# Benchmark de performance
python src/tests/test_system.py --benchmark
```

### Teste Interativo

```bash
# Jupyter notebooks
jupyter notebook notebooks/

# Chat interativo
python -c "from src.agent.eco_travel_agent import EcoTravelAgent; EcoTravelAgent().chat()"
```

## Configurações Avançadas

### Personalizar Sistema RAG

```python
# src/config/rag_config.py
RAG_CONFIG = {
    "chunk_size": 512,
    "chunk_overlap": 50,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "top_k": 10,
    "hybrid_alpha": 0.5  # Peso da busca semântica vs lexical
}
```

### Configurar Modelos de Embedding

```bash
# Modelos alternativos (melhores para português)
pip install sentence-transformers

# No código Python:
rag = AdvancedRAGSystem(
    embedding_model="neuralmind/bert-base-portuguese-cased"
)
```

### Otimizações de Performance

```bash
# Para sistemas com GPU
pip install faiss-gpu torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Para sistemas ARM (Apple Silicon)
conda install faiss-cpu pytorch torchvision torchaudio -c pytorch
```

## Estrutura de Arquivos Após Instalação

```
ecotravel-agent/
├── data/                    # Base de conhecimento
│   ├── guias/              # Guias de viagem
│   ├── emissoes/           # Dados de emissões
│   └── avaliacoes/         # Avaliações de hotéis
├── src/                    # Código fonte
│   ├── agent/              # Agente principal
│   ├── rag/                # Sistema RAG
│   ├── tools/              # Ferramentas customizadas
│   └── tests/              # Testes e benchmarks
├── notebooks/              # Jupyter notebooks
├── docs/                   # Documentação
├── requirements.txt        # Dependências
├── .env                    # Configurações
└── README.md
```

## Solução de Problemas

### Problemas Comuns

#### 1. Erro de Importação

```bash
# Problema: ModuleNotFoundError
# Solução: Verificar PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Ou no Windows
set PYTHONPATH=%PYTHONPATH%;%cd%\src
```

#### 2. Erro de Memória

```python
# Reduzir uso de memória
agent = EcoTravelAgent(
    chunk_size=256,  # Reduzir chunks
    top_k=5          # Menos resultados RAG
)
```

#### 3. LLM Não Disponível

```bash
# Verificar status do Ollama
ollama list
ollama ps

# Restart se necessário
ollama serve
```

#### 4. Erro de Rede/API

```python
# Configurar timeouts
import os
os.environ['REQUESTS_TIMEOUT'] = '30'

# Usar apenas ferramentas offline
agent = EcoTravelAgent(offline_mode=True)
```

### Logs de Debug

```python
# Ativar logs detalhados
import logging
logging.basicConfig(level=logging.DEBUG)

# Executar com verbose
agent = EcoTravelAgent(verbose=True)
```

### Performance

```bash
# Monitorar uso de recursos
htop          # Linux/macOS
taskmgr       # Windows

# Profiling de performance
python -m cProfile src/agent/eco_travel_agent.py
```

## Atualizações

### Atualizar Código

```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

### Atualizar Modelos

```bash
# Ollama
ollama pull llama3

# Sentence Transformers (automático na primeira execução)
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

### Atualizar Base de Conhecimento

```bash
# Reconstruir índices RAG após adicionar novos dados
python -c "from src.rag.rag_system import AdvancedRAGSystem; rag = AdvancedRAGSystem(); rag.build_index()"
```

## Instalação para Desenvolvimento

### Dependências Adicionais

```bash
# Ferramentas de desenvolvimento
pip install black flake8 pytest mypy pre-commit

# Configurar pre-commit hooks
pre-commit install
```

### Executar Testes

```bash
# Testes unitários
pytest src/tests/

# Testes de integração
python src/tests/test_system.py --tests

# Coverage
pytest --cov=src src/tests/
```

### Documentação

```bash
# Gerar documentação
pip install sphinx sphinx-rtd-theme
cd docs/
make html
```

## Suporte

### Recursos de Ajuda

- **Documentação**: `docs/`
- **Exemplos**: `notebooks/`
- **Testes**: `src/tests/`
- **Issues**: GitHub Issues

### Canais de Suporte

1. **GitHub Issues**: Para bugs e feature requests
2. **Discussions**: Para dúvidas gerais
3. **Wiki**: Para tutoriais e exemplos

### Relatando Problemas

Inclua sempre:

```bash
# Informações do sistema
python --version
pip list | grep -E "(langchain|torch|transformers)"

# Logs de erro
python src/tests/test_system.py --report 2>&1 | tee debug.log
```

---

**🎉 Parabéns! O EcoTravel Agent está pronto para uso!**

Execute `python -c "from src.agent.eco_travel_agent import EcoTravelAgent; EcoTravelAgent().chat()"` para começar a planejar viagens sustentáveis!