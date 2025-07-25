# 🚀 Guia de Execução Rápida - EcoTravel Agent

## 🎯 Opções de Execução

### 1. 🔥 EXECUÇÃO MAIS RÁPIDA - Google Colab

**⏱️ Tempo: 5 minutos para estar rodando**

1. **Abra o Colab**: [notebooks/EcoTravel_Agent_RL_Colab.ipynb](notebooks/EcoTravel_Agent_RL_Colab.ipynb)

2. **Configure API Keys** (OBRIGATÓRIO):
   ```
   🔑 No Colab: Clique no ícone de chave na barra lateral esquerda
   
   Adicione:
   - OPENAI_API_KEY: sua_chave_openai_aqui
   - GOOGLE_API_KEY: sua_chave_google_aqui (opcional)
   ```

3. **Execute todas as células**: `Runtime > Run all`

4. **Aguarde o treinamento** (2-3 minutos) e veja os resultados!

---

### 2. 💻 EXECUÇÃO LOCAL - Desenvolvimento

**⏱️ Tempo: 10-15 minutos para setup completo**

```bash
# Clone e setup
git clone <repo-url>
cd projeto-ecotravel
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt

# Configure APIs
export OPENAI_API_KEY="sua-chave"
export GOOGLE_API_KEY="sua-chave"  # opcional

# Execute demo rápido
python demo_ecotravel.py

# OU execute sistema completo
python src/agent/ecotravel_agent_rl.py
```

---

### 3. 🐳 EXECUÇÃO COM DOCKER

**⏱️ Tempo: 5 minutos após build**

```bash
# Build e execute
docker build -t ecotravel .
docker run -it --rm -e OPENAI_API_KEY="sua-chave" ecotravel
```

---

## 🔑 APIs Necessárias

### Obrigatória:
- **OpenAI**: [platform.openai.com](https://platform.openai.com/api-keys)
  - Crie conta gratuita
  - Gere API key
  - Use nos notebooks/código

### Opcionais:
- **Google Search**: [console.developers.google.com](https://console.developers.google.com)
- **DeepSeek**: [platform.deepseek.com](https://platform.deepseek.com)

---

## ✅ Checklist de Verificação

Antes de executar, verifique:

- [ ] **Python 3.8+** instalado
- [ ] **OpenAI API Key** configurada
- [ ] **Dependências** instaladas (`pip install -r requirements.txt`)
- [ ] **Ambiente virtual** ativo (recomendado)

---

## 🧪 Teste Rápido de Funcionamento

### No Colab:
Execute a célula de teste no notebook - deve mostrar:
```
✅ Dependências instaladas!
✅ APIs configuradas!
🚀 Sistema RL treinado em 2000 timesteps
🎯 RL recomenda: RAG (confiança: 85%)
```

### Local:
```bash
python -c "
from src.rag.rag_system import AdvancedRAGSystem
rag = AdvancedRAGSystem()
print('✅ Sistema funcionando!')
"
```

---

## 🚨 Problemas Comuns

### "ModuleNotFoundError"
```bash
# Solução:
pip install -r requirements.txt --upgrade
```

### "API Key inválida"
```bash
# Verificar:
echo $OPENAI_API_KEY
# Deve mostrar: sk-...
```

### "Erro de memória no Colab"
```python
# No notebook, usar:
agent = EcoTravelRLAgent(use_advanced_embeddings=False)
```

---

## 📱 Que Esperar

### Primeira Execução:
1. **Download de modelos** (2-3 min)
2. **Treinamento RL** (2-5 min)
3. **Teste de ferramentas** (1 min)
4. **Demonstrações interativas**

### Funcionalidades:
- ✅ **Cálculo de CO2** por viagem
- ✅ **Recomendações sustentáveis** baseadas em RL
- ✅ **Dashboard de métricas** em tempo real
- ✅ **Comparação de performance** antes/depois RL

---

## 🎯 Exemplo de Uso Imediato

```python
# Cole este código no Colab após executar setup:

query = "Quero viajar de São Paulo para o Rio sustentavelmente"
response, metrics = agent.process_query(query)

print(f"🎯 RL escolheu: {metrics['rl_recommendation']['recommended_tool']}")
print(f"💬 Resposta: {response}")
```

**Resultado esperado:**
```
🎯 RL escolheu: RAG (confiança: 85%)
💬 Resposta: Para viagem SP-RJ sustentável:
- Ônibus: 35.6 kg CO2, economia de 82% vs. avião
- Hotel Verde Rio: certificação LEED Gold
- Atividades: trilhas ecológicas, turismo responsável
```

---

## 🏆 Demonstração dos Diferenciais

Após execução, você verá:

1. **RL em ação**: Seleção inteligente de ferramentas
2. **RAG avançado**: Busca híbrida com reranking
3. **Métricas reais**: Performance quantificada
4. **Dashboard interativo**: Visualizações em tempo real
5. **Sustentabilidade**: Cálculos de CO2 e alternativas eco-friendly

---

**⚡ Em menos de 5 minutos no Colab você terá um sistema de IA avançado funcionando com RL + RAG + Multi-tool!**