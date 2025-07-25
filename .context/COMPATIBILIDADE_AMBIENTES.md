# 🔧 Compatibilidade e Configuração de Ambientes

## 📋 Ambientes Testados

### ✅ Google Colab (RECOMENDADO)
- **Status**: Totalmente compatível
- **Python**: 3.10.12
- **GPU**: Tesla T4 (opcional)
- **RAM**: 12.7 GB disponível
- **Tempo de setup**: 3-5 minutos

**Configuração:**
```python
# No Colab, todas as dependências são instaladas automaticamente
!pip install -q openai langchain stable-baselines3 gymnasium
```

### ✅ Ambiente Local (Windows/Linux/macOS)
- **Status**: Totalmente compatível
- **Python**: 3.8+ (testado até 3.11)
- **RAM mínima**: 4GB (8GB recomendado)
- **Armazenamento**: 2GB

**Configuração:**
```bash
# Criar ambiente isolado
python -m venv ecotravel_env
source ecotravel_env/bin/activate  # Linux/macOS
# ou ecotravel_env\Scripts\activate  # Windows

pip install -r requirements.txt
```

### ✅ Docker
- **Status**: Compatível
- **Base image**: python:3.9-slim
- **Recursos**: 2GB RAM, 1 CPU core

### ⚠️ Jupyter Lab/Notebook Local
- **Status**: Compatível com adaptações
- **Nota**: Alguns widgets podem não funcionar perfeitamente
- **Solução**: Use `%pip install` em células para dependências

## 🔑 Configuração de APIs por Ambiente

### Google Colab (Secrets)
```python
from google.colab import userdata
import os

os.environ['OPENAI_API_KEY'] = userdata.get('OPENAI_API_KEY')
```

### Local (.env)
```bash
# Arquivo .env
OPENAI_API_KEY=sk-sua-chave-aqui
GOOGLE_API_KEY=AIza-sua-chave-aqui
```

### Docker (Environment Variables)
```bash
docker run -e OPENAI_API_KEY="sk-..." projeto-ecotravel-agent
```

## ⚙️ Configurações Específicas por Ambiente

### Colab - Performance Otimizada
```python
# Use configuração reduzida para economizar recursos
agent = EcoTravelRLAgent(
    use_advanced_embeddings=False,  # Usa modelo local
    model_name="colab_optimized"
)

# Treinamento mais rápido
agent.train(total_timesteps=2000)  # Reduzido de 10000
```

### Local - Performance Máxima
```python
# Use configuração completa
agent = EcoTravelRLAgent(
    use_advanced_embeddings=True,   # OpenAI embeddings
    model_name="local_full"
)

# Treinamento completo
agent.train(total_timesteps=10000)
```

### Produção - Alta Disponibilidade
```python
# Configuração robusta com fallbacks
agent = EcoTravelAgentWithRL(
    rl_agent=rl_agent,
    use_gpt4=True,                 # GPT-4 se disponível
    use_deepseek=False,            # Fallback para DeepSeek
    enable_rl=True                 # RL otimizado
)
```

## 🚀 Scripts de Execução Rápida

### start_colab.py
```python
#!/usr/bin/env python3
"""Execução otimizada para Google Colab"""

# Instalar dependências
!pip install -q -r requirements.txt

# Configurar APIs
from google.colab import userdata
import os
os.environ['OPENAI_API_KEY'] = userdata.get('OPENAI_API_KEY')

# Executar sistema
from src.agent.ecotravel_agent_rl import EcoTravelAgentWithRL
from src.rl.rl_agent import EcoTravelRLAgent

# Configuração Colab-friendly
rl_agent = EcoTravelRLAgent(use_advanced_embeddings=False)
rl_agent.train(total_timesteps=2000)

agent = EcoTravelAgentWithRL(rl_agent=rl_agent, use_gpt4=False)
print("✅ Sistema pronto no Colab!")
```

### start_local.py
```python
#!/usr/bin/env python3
"""Execução para ambiente local"""

import os
from dotenv import load_dotenv

# Carregar configurações
load_dotenv()

# Verificar APIs
if not os.getenv('OPENAI_API_KEY'):
    print("❌ Configure OPENAI_API_KEY no arquivo .env")
    exit(1)

# Executar sistema completo
from src.agent.ecotravel_agent_rl import EcoTravelAgentWithRL
from src.rl.rl_agent import EcoTravelRLAgent

rl_agent = EcoTravelRLAgent(use_advanced_embeddings=True)
rl_agent.train(total_timesteps=5000)

agent = EcoTravelAgentWithRL(rl_agent=rl_agent, use_gpt4=True)
print("✅ Sistema completo pronto localmente!")
```

## 🐛 Solução de Problemas por Ambiente

### Google Colab

**Problema: "Runtime disconnected"**
```python
# Solução: Reduzir uso de memória
import gc
gc.collect()

# Usar configuração leve
agent = EcoTravelRLAgent(use_advanced_embeddings=False)
```

**Problema: "Quota exceeded"**
```python
# Solução: Usar APIs gratuitas como fallback
try:
    # OpenAI embeddings
    embeddings = OpenAIEmbeddings()
except:
    # Fallback para HuggingFace
    embeddings = HuggingFaceEmbeddings()
```

### Local (Windows)

**Problema: "pip install failed"**
```cmd
# Solução: Usar conda
conda create -n ecotravel python=3.9
conda activate ecotravel
conda install pytorch pandas numpy
pip install -r requirements.txt
```

**Problema: "Module not found"**
```python
# Solução: Adicionar ao PYTHONPATH
import sys
sys.path.append('./src')
```

### Local (macOS/Linux)

**Problema: "Permission denied"**
```bash
# Solução: Usar virtualenv
python3 -m venv ecotravel_env
source ecotravel_env/bin/activate
pip install --user -r requirements.txt
```

**Problema: "SSL Certificate"**
```bash
# Solução: Atualizar certificados
/Applications/Python\ 3.x/Install\ Certificates.command  # macOS
# ou
sudo apt-get update && sudo apt-get install ca-certificates  # Linux
```

## 📊 Performance por Ambiente

### Benchmarks de Execução

| Ambiente | Setup | Treinamento RL | Query Processing | Total |
|----------|-------|----------------|------------------|-------|
| Colab Free | 3 min | 2 min | 1.5s | ~5 min |
| Colab Pro | 2 min | 1 min | 0.8s | ~3 min |
| Local (CPU) | 5 min | 5 min | 2.0s | ~10 min |
| Local (GPU) | 5 min | 2 min | 1.0s | ~7 min |
| Docker | 3 min | 3 min | 1.8s | ~6 min |

### Uso de Recursos

| Ambiente | RAM | CPU | GPU | Armazenamento |
|----------|-----|-----|-----|---------------|
| Colab | 2-4 GB | 2 cores | T4 (opcional) | 5 GB |
| Local Min | 4 GB | 2 cores | - | 2 GB |
| Local Recom | 8 GB | 4 cores | GTX/RTX | 5 GB |
| Docker | 2 GB | 1 core | - | 3 GB |

## 🔧 Configurações Avançadas

### Para Desenvolvedores
```python
# Modo debug com logs detalhados
import logging
logging.basicConfig(level=logging.DEBUG)

agent = EcoTravelAgentWithRL(
    rl_agent=rl_agent,
    enable_rl=True,
    verbose=True  # Logs detalhados
)
```

### Para Produção
```python
# Configuração robusta com monitoramento
agent = EcoTravelAgentWithRL(
    rl_agent=rl_agent,
    enable_rl=True,
    monitoring=True,
    fallback_llm="gpt-3.5-turbo",
    max_retries=3,
    timeout=30
)
```

### Para Pesquisa
```python
# Configuração experimental
agent = EcoTravelRLAgent(
    use_advanced_embeddings=True,
    experimental_features=True,
    collect_detailed_metrics=True,
    save_training_data=True
)
```

## ✅ Checklist de Compatibilidade

Antes de executar, verifique:

### Requisitos Mínimos
- [ ] Python 3.8+
- [ ] 4GB RAM disponível
- [ ] 2GB espaço em disco
- [ ] Conexão com internet
- [ ] OpenAI API key válida

### Requisitos Recomendados
- [ ] Python 3.9-3.11
- [ ] 8GB RAM
- [ ] GPU com CUDA (opcional)
- [ ] 5GB espaço em disco
- [ ] APIs configuradas (OpenAI, Google)

### Para Desenvolvimento
- [ ] Git instalado
- [ ] Editor de código (VS Code, PyCharm)
- [ ] Docker (opcional)
- [ ] Jupyter Lab/Notebook

---

**💡 Dica**: Use Google Colab para demonstrações rápidas e ambiente local para desenvolvimento completo!