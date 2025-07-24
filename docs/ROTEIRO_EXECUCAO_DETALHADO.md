# 📋 Roteiro de Execução Detalhado - EcoTravel Agent com RL

## 🎯 Visão Geral do Projeto

O EcoTravel Agent é um sistema avançado que combina:
- **Reinforcement Learning (RL)** para otimização de seleção de ferramentas
- **RAG avançado** com estratégias anti-alucinação
- **Multi-tool orchestration** com 4 ferramentas integradas
- **Foco em sustentabilidade** e redução de CO2

## 📅 Cronograma de Execução (4 Semanas)

### Semana 1: Preparação e Setup

#### Dia 1-2: Configuração do Ambiente
```bash
# 1. Criar estrutura de pastas
mkdir -p projeto-ecotravel/{src/{rl,agent,dashboard},data/{guias,emissoes,avaliacoes},notebooks,models,metrics,docs}

# 2. Configurar ambiente Python
conda create -n ecotravel python=3.10
conda activate ecotravel

# 3. Instalar dependências base
pip install -r requirements.txt
```

#### Dia 3-4: Coleta e Preparação de Dados
- [ ] Baixar guias de viagem sustentáveis (PDFs)
- [ ] Criar tabelas de emissões de CO2
- [ ] Coletar avaliações de hotéis eco-friendly
- [ ] Estruturar dados em formato JSON/CSV

```python
# Script para preparar dados
import pandas as pd

# Tabela de emissões
emissoes_df = pd.DataFrame({
    'transporte': ['aviao', 'carro', 'onibus', 'trem'],
    'co2_kg_por_km': [0.255, 0.171, 0.089, 0.041]
})
emissoes_df.to_csv('data/emissoes/tabela_emissoes.csv')
```

#### Dia 5: Setup de APIs
- [ ] Obter chaves de API (OpenAI, Google, HuggingFace, DeepSeek)
- [ ] Configurar variáveis de ambiente
- [ ] Testar conectividade

```bash
# .env file
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="AIza..."
export HF_TOKEN="hf_..."
export DEEPSEEK_API_KEY="sk-..."
```

### Semana 2: Implementação Core

#### Dia 6-7: Ambiente RL
Implementar `src/rl/environment.py`:

```python
class EcoTravelRLEnvironment(gym.Env):
    def __init__(self):
        # Estado: embeddings + features
        # Ação: escolha de ferramenta
        # Recompensa: multi-objetivo
```

**Checklist:**
- [ ] Definir espaço de observação (embeddings 1536D)
- [ ] Definir espaço de ação (4 ferramentas)
- [ ] Implementar função de recompensa
- [ ] Testar ambiente com ações aleatórias

#### Dia 8-9: Agente PPO
Implementar `src/rl/rl_agent.py`:

```python
class EcoTravelRLAgent:
    def __init__(self):
        self.model = PPO("MlpPolicy", env, ...)
    
    def train(self, timesteps=10000):
        self.model.learn(timesteps)
```

**Checklist:**
- [ ] Configurar hiperparâmetros PPO
- [ ] Implementar callbacks de checkpoint
- [ ] Criar métricas de treinamento
- [ ] Treinar modelo inicial (5k steps)

#### Dia 10: Sistema RAG
Implementar RAG avançado:

```python
# Estratégias modernas
- Hybrid search (BM25 + Semantic)
- Reranking
- Multi-query retriever
- Anti-hallucination prompts
```

**Checklist:**
- [ ] Configurar chunking inteligente
- [ ] Implementar ensemble retriever
- [ ] Adicionar verificação de fontes
- [ ] Testar com queries de exemplo

### Semana 3: Integração e Otimização

#### Dia 11-12: Integração LangChain + RL
Implementar `src/agent/ecotravel_agent_rl.py`:

```python
class EcoTravelAgentWithRL:
    def process_query(self, query):
        # 1. RL prediz ferramenta
        # 2. Agent executa com recomendação
        # 3. Coleta métricas
```

**Checklist:**
- [ ] Conectar RL com seleção de ferramentas
- [ ] Implementar todas as 4 ferramentas
- [ ] Adicionar logging de métricas
- [ ] Testar pipeline completo

#### Dia 13-14: Dashboard de Métricas
Implementar `src/dashboard/metrics_dashboard.py`:

**Features do Dashboard:**
- [ ] Visualização de distribuição de ferramentas
- [ ] Gráficos de evolução do treinamento
- [ ] Comparação antes/depois do RL
- [ ] Métricas de sustentabilidade

#### Dia 15: Otimização e Testes
- [ ] Treinar modelo RL por mais timesteps (20k+)
- [ ] Ajustar hiperparâmetros baseado em métricas
- [ ] Implementar testes unitários
- [ ] Validar com casos de uso reais

### Semana 4: Finalização e Entrega

#### Dia 16-17: Notebook Colab
Criar notebook completo com:
- [ ] Setup e instalação automatizada
- [ ] Demonstração passo a passo
- [ ] Visualizações interativas
- [ ] Casos de uso documentados

#### Dia 18: Documentação
- [ ] README.md detalhado
- [ ] Fluxograma da arquitetura (Draw.io)
- [ ] Documento PDF final
- [ ] Vídeo demonstrativo (opcional)

#### Dia 19-20: Testes Finais e Entrega
- [ ] Teste completo no Colab
- [ ] Validar todas as dependências
- [ ] Preparar apresentação
- [ ] Submeter projeto

## 🛠️ Comandos Úteis

### Treinar Agente RL
```bash
python -m src.rl.rl_agent --train --timesteps 10000
```

### Executar Agente Completo
```bash
python src/agent/ecotravel_agent_rl.py
```

### Iniciar Dashboard
```bash
streamlit run src/dashboard/metrics_dashboard.py
```

### Executar Demo
```bash
python demo_ecotravel.py
```

## 📊 Métricas de Sucesso

1. **Performance RL**
   - Taxa de acerto > 85%
   - Tempo de resposta < 2s
   - Redução de alucinações > 15%

2. **Qualidade RAG**
   - Hit Rate > 0.8
   - MRR > 0.7
   - Latência < 500ms

3. **Sustentabilidade**
   - Recomendações priorizando baixo CO2
   - Cálculos precisos de emissões
   - Alternativas eco-friendly sugeridas

## 🚨 Troubleshooting

### Problema: Erro de GPU no Colab
```python
# Solução: Usar CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
```

### Problema: API Key inválida
```python
# Solução: Verificar secrets do Colab
from google.colab import userdata
os.environ['OPENAI_API_KEY'] = userdata.get('OPENAI_API_KEY')
```

### Problema: Memória insuficiente
```python
# Solução: Reduzir batch size e embedding dim
batch_size = 32  # ao invés de 64
use_advanced_embeddings = False  # usar modelo menor
```

## ✅ Checklist Final

- [ ] Código completo e funcional
- [ ] Documentação clara e detalhada
- [ ] Notebook executável no Colab
- [ ] Métricas demonstrando melhorias
- [ ] Fluxograma visual da arquitetura
- [ ] README com instruções de uso
- [ ] Demonstração de caso de uso real

## 🎯 Dicas para Superar Outros Alunos

1. **Inovação Técnica**: RL + LangChain é combinação única
2. **Métricas Quantitativas**: Apresente números concretos
3. **UI Profissional**: Dashboard interativo impressiona
4. **Código Limpo**: Documentação e modularidade
5. **Aplicação Real**: Foco em sustentabilidade é diferencial

## 📞 Suporte

Em caso de dúvidas durante a implementação:
1. Consulte a documentação das bibliotecas
2. Use o modo debug verbose
3. Verifique logs de erro detalhados
4. Teste componentes isoladamente

---

**Boa sorte com a implementação! 🚀** 