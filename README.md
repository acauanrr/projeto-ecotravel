# 🌍 EcoTravel Agent - Sistema Inteligente de Viagens Sustentáveis com RL

**Trabalho Prático 5 - Agentes com LLMs**  
**Disciplina:** PPGINF528 - Tópicos Especiais em Recuperação da Informação - NLP
**Professores:** Prof. Dr. André Carvalho e Prof. Dr. Altigran da Silva
**Universidade Federal do Amazonas (UFAM)**  
**Disciplina:** Inteligência Artificial Avançada

## 👥 Aluno:

- **Acauan Cardoso Ribeiro** - Matrícula: 3240232


## 📋 Visão Geral do Projeto

O **EcoTravel Agent** é um sistema inteligente que combina **Reinforcement Learning**, **RAG (Retrieval-Augmented Generation)** e **múltiplas ferramentas** para planejar viagens sustentáveis. O sistema utiliza um agente baseado em LLM que aprende a selecionar otimamente entre diferentes ferramentas para responder queries sobre turismo ecológico.

### 🎯 Objetivo Principal

Desenvolver um agente inteligente capaz de:
- Processar consultas sobre viagens sustentáveis
- Selecionar automaticamente a ferramenta mais adequada (RAG, APIs, Calculadoras)
- Fornecer respostas precisas sobre destinos eco-friendly e pegada de carbono
- Aprender e otimizar suas decisões através de Reinforcement Learning

### 🔧 Tecnologias Utilizadas

- **LLM Base**: OpenAI GPT (via LangChain)
- **Reinforcement Learning**: PPO (Proximal Policy Optimization) com Stable-Baselines3
- **RAG System**: Sentence Transformers + FAISS + BM25
- **Ferramentas**: APIs de clima, calculadora de CO2, busca web
- **Ambiente**: Gymnasium customizado
- **Visualização**: Matplotlib, Plotly, Streamlit

## 🚀 Guia de Execução para o Professor

### ⚡ **Execução Rápida (5 minutos)**

```bash
# 1. Clonar o repositório
git clone <url-do-repositorio>
cd projeto-ecotravel

# 2. Criar ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# OU
.venv\Scripts\activate     # Windows

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Executar o notebook principal
jupyter notebook EcoTravel_Agent_RL_Local_Completo.ipynb
```

### 📋 **Pré-requisitos**

- **Python**: 3.8 ou superior
- **Memória RAM**: Mínimo 8GB (recomendado 16GB)
- **Espaço em disco**: 2GB livres
- **GPU**: Opcional (CUDA compatível para aceleração)

### 🔧 **Instalação Detalhada**

#### Passo 1: Preparação do Ambiente

```bash
# Verificar versão do Python
python --version  # Deve ser 3.8+

# Criar ambiente virtual isolado
python -m venv .venv

# Ativar ambiente virtual
# Linux/Mac:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# Atualizar pip
pip install --upgrade pip
```

#### Passo 2: Instalação de Dependências

```bash
# Instalar todas as dependências do projeto
pip install -r requirements.txt

# Verificar instalação das principais bibliotecas
python -c "
import langchain
import sentence_transformers
import gymnasium
import stable_baselines3
import matplotlib
import pandas
print('✅ Todas as dependências instaladas com sucesso!')
"
```

#### Passo 3: Configuração de APIs (Opcional)

```bash
# Criar arquivo .env para APIs (opcional - o sistema funciona sem)
echo 'OPENAI_API_KEY=sua_chave_aqui' > .env
echo 'GOOGLE_API_KEY=sua_chave_aqui' >> .env
```

**Nota**: O sistema funciona completamente **sem APIs externas**. As chaves são opcionais para funcionalidades avançadas.

### 📓 **Executando o Notebook Principal**

#### Opção 1: Jupyter Notebook (Recomendado)

```bash
# Instalar Jupyter se não tiver
pip install jupyter

# Iniciar Jupyter
jupyter notebook

# Abrir: EcoTravel_Agent_RL_Local_Completo.ipynb
```

#### Opção 2: Google Colab

1. Fazer upload do arquivo `EcoTravel_Agent_RL_Local_Completo.ipynb`
2. Executar a primeira célula para instalar dependências
3. Executar as células sequencialmente

#### Opção 3: VS Code

```bash
# Instalar extensão Python + Jupyter no VS Code
# Abrir o arquivo .ipynb
# Executar células com Ctrl+Enter
```

### 🧪 **Teste de Verificação**

Execute este comando para verificar se tudo está funcionando:

```bash
python -c "
import sys
sys.path.append('.')

# Teste básico do sistema
from src.rag.enhanced_rag_system import EnhancedRAGSystem
from src.tools.carbon_calculator import CarbonCalculator
from src.rl.environment import EcoTravelRLEnvironment

print('🧪 Testando componentes...')

# Teste RAG
rag = EnhancedRAGSystem()
rag.load_data()
print('✅ Sistema RAG funcionando')

# Teste Calculadora
calc = CarbonCalculator()
result = calc.calculate_carbon_footprint('aviao_domestico', 500, True)
print(f'✅ Calculadora CO2: {result[\"total_emissions_kg\"]} kg')

# Teste Ambiente RL
env = EcoTravelRLEnvironment(use_advanced_embeddings=False)
state, _ = env.reset()
print(f'✅ Ambiente RL: {state.shape} dimensões')

print('🎉 Todos os componentes funcionando!')
"
```

## 📊 Estrutura do Projeto

```
projeto-ecotravel/
├── 📓 EcoTravel_Agent_RL_Local_Completo.ipynb  # NOTEBOOK PRINCIPAL
├── 📂 src/                                      # Código fonte
│   ├── rag/
│   │   └── enhanced_rag_system.py              # Sistema RAG avançado
│   ├── tools/
│   │   ├── carbon_calculator.py                # Calculadora de CO2
│   │   ├── weather_api.py                      # API de clima
│   │   └── web_search.py                       # Busca web
│   ├── rl/
│   │   ├── environment.py                      # Ambiente Gymnasium
│   │   └── rl_agent.py                         # Agente PPO
│   └── agent/
│       └── ecotravel_agent_rl.py               # Agente principal
├── 📂 data/                                    # Base de conhecimento
│   ├── guias/                                  # Guias de turismo sustentável
│   ├── emissoes/                               # Dados de emissões CO2
│   └── avaliacoes/                             # Avaliações de hotéis
├── 📂 docs/                                    # Documentação
├── 📄 requirements.txt                         # Dependências Python
└── 📄 README.md                               # Este arquivo
```

## 🎯 Como Avaliar o Projeto

### 1. **Execução do Notebook** (40 pontos)

Execute o notebook `EcoTravel_Agent_RL_Local_Completo.ipynb` célula por célula:

- **Células 1-4**: Instalação e configuração (deve executar sem erros)
- **Células 5-6**: Testes de integração (verificar se componentes funcionam)
- **Células 7-10**: Treinamento do agente RL (observar métricas de aprendizado)
- **Células 11-14**: Sistema integrado e demonstrações (testar queries)
- **Células 15-18**: Análise de performance e relatórios

### 2. **Funcionalidades Principais** (30 pontos)

Teste as seguintes funcionalidades:

```python
# No notebook, execute:
result = agent.process_query("Quais são os destinos sustentáveis no Nordeste?")
print(result)

# Deve retornar informações relevantes usando o sistema RAG
```

```python
# Teste da calculadora de CO2:
result = agent.process_query("Calcule emissões de um voo São Paulo - Salvador")
print(result)

# Deve calcular e retornar emissões em kg de CO2
```

### 3. **Reinforcement Learning** (20 pontos)

Observe na célula 10 do notebook:
- Treinamento do agente PPO
- Melhoria das métricas ao longo do tempo
- Seleção inteligente de ferramentas

### 4. **Qualidade do Código** (10 pontos)

- Código bem estruturado e documentado
- Tratamento de erros robusto
- Arquitetura modular

## 📈 Resultados Esperados

Ao executar o notebook, você deve observar:

1. **Sistema RAG**: Busca informações em guias de turismo sustentável
2. **Calculadora CO2**: Calcula emissões para diferentes transportes
3. **Agente RL**: Aprende a selecionar a ferramenta mais adequada
4. **Integração**: Sistema completo respondendo queries complexas

### Exemplo de Saída:

```
🔍 Processando: 'Quais são os destinos sustentáveis no Nordeste?'
📋 Tipo identificado: travel
🛠️ Ferramenta selecionada: rag

📊 Resultado (Ferramenta: rag):
✓ Contexto encontrado: Fernando de Noronha oferece turismo sustentável com...
✓ Fontes consultadas: 5 documentos
```

## 🔧 Solução de Problemas

### Problema: Erro de dependência

```bash
# Solução:
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### Problema: Jupyter não encontra kernel

```bash
# Solução:
python -m ipykernel install --user --name=.venv
```

### Problema: Erro de memória

```bash
# Solução: Reduzir batch size no notebook
# Modificar na célula 10: batch_size=32 para batch_size=16
```

### Problema: GPU não detectada

```bash
# Verificar CUDA:
python -c "import torch; print(torch.cuda.is_available())"

# Se False, o sistema funcionará em CPU (mais lento, mas funcional)
```

## 📚 Documentação Adicional

- **Relatório Técnico**: Veja `docs/relatorio_tecnico.pdf`
- **Arquitetura**: Documentação em `docs/arquitetura.md`
- **API Reference**: Documentação das classes em `docs/api/`

## 🏆 Critérios de Avaliação Atendidos

✅ **Implementação de Agente com LLM**: Sistema completo com LangChain  
✅ **Múltiplas Ferramentas**: RAG, APIs, Calculadoras integradas  
✅ **Reinforcement Learning**: PPO para otimização de seleção  
✅ **Base de Conhecimento**: RAG com dados reais de turismo sustentável  
✅ **Métricas e Avaliação**: Dashboard completo com visualizações  
✅ **Código Limpo**: Bem estruturado e documentado  
✅ **Funcionalidade Completa**: Sistema end-to-end operacional  

## 📞 Suporte

Em caso de dúvidas na execução:

1. **Primeiro**: Verifique se todas as dependências estão instaladas
2. **Segundo**: Execute o teste de verificação acima
3. **Terceiro**: Consulte a seção "Solução de Problemas"
4. **Último recurso**: Contate os desenvolvedores

---

**🌍 Sistema completo de viagens sustentáveis com IA - Pronto para avaliação!**