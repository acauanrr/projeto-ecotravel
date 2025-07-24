# 📋 Instruções de Organização do Projeto

## 🎯 Diretrizes Gerais

### Estrutura de Arquivos
- **NUNCA criar novos arquivos na raiz** do projeto, salvo raras exceções
- **TODO arquivo deve se encaixar** na estrutura de pastas existente
- **Preferir sempre editar** arquivos existentes ao invés de criar novos
- **NUNCA criar documentação proativa** (*.md) ou README files sem solicitação explícita

### Pasta .context/
- **Contém documentação** centralizada do projeto
- **Verificar sempre** se existe arquivo similar antes de criar novo
- **Atualizar arquivos existentes** ao invés de duplicar conteúdo
- **Manter padrão** de nomenclatura e formatação

### Pasta setup/
- **Scripts de instalação** e configuração
- **Demos e testes** de funcionalidade
- **Verificar e corrigir** arquivos existentes antes de criar novos
- **Manter projeto limpo** e organizado para execução

## 📁 Estrutura Atual Organizada

```
trab_5-agents/
├── .context/                    # Documentação centralizada
│   ├── 01_DESCRICAO_TRABALHO.md      # Descrição original do TP5
│   ├── PROJETO_ANALISE_REVISAO.md    # Análise completa + arquitetura
│   ├── GUIA_EXECUCAO_RAPIDA.md       # Guia de execução rápida
│   ├── COMPATIBILIDADE_AMBIENTES.md  # Configurações por ambiente
│   ├── PREPARACAO_GITHUB.md          # Preparação para GitHub
│   ├── RESUMO_REVISAO_FINAL.md       # Resumo da revisão final
│   └── 00_INSTRUCOES_ORGANIZACAO.md  # Este arquivo (meta)
├── src/                         # Código fonte principal
│   ├── agent/                   # Agentes principais
│   ├── rag/                     # Sistema RAG
│   ├── rl/                      # Reinforcement Learning
│   ├── tools/                   # Ferramentas customizadas
│   ├── dashboard/               # Dashboard Streamlit
│   └── tests/                   # Testes e benchmarks
├── setup/                       # Scripts de configuração
│   ├── demo_ecotravel.py        # Demo simplificado
│   ├── install_dependencies.py  # Instalação de dependências
│   ├── setup.py                 # Configuração do pacote
│   └── test_installation.py     # Teste de instalação
├── data/                        # Dados do projeto
├── docs/                        # Documentação técnica
├── notebooks/                   # Jupyter notebooks
└── [arquivos raiz]              # README, requirements, etc.
```

## ✅ Checklist de Organização

### Antes de Criar Novo Arquivo:
- [ ] Verificar se existe arquivo similar na estrutura
- [ ] Tentar atualizar arquivo existente primeiro
- [ ] Confirmar que a pasta de destino está correta
- [ ] Verificar padrão de nomenclatura

### Para Relatórios e Documentação:
- [ ] **Sempre salvar na pasta .context/**
- [ ] Verificar se não existe documento similar
- [ ] Atualizar documento existente se aplicável
- [ ] Seguir padrão de nomenclatura (XX_NOME_DOCUMENTO.md)

### Para Scripts e Código:
- [ ] **Sempre salvar na pasta setup/** ou **src/**
- [ ] Verificar funcionalidade dos scripts existentes
- [ ] Corrigir problemas antes de criar novos
- [ ] Manter compatibilidade com estrutura existente

## 🔄 Processo de Manutenção

### Quando Solicitado Análise:
1. **Analisar estrutura completa** (pastas e arquivos)
2. **Revisar pasta .context** e identificar redundâncias
3. **Revisar pasta setup** e corrigir problemas
4. **Padronizar** arquivos duplicados/redundantes
5. **Documentar** no arquivo de instruções (este)

### Padrão de Atualização:
- **Consolidar** conteúdo similar em arquivo único
- **Remover** arquivos redundantes após consolidação
- **Manter** estrutura hierárquica clara
- **Preservar** informações importantes durante limpeza

## 📝 Convenções de Nomenclatura

### Pasta .context/
- `00_` - Meta-informações e instruções
- `01-99_` - Documentos numerados por ordem lógica
- `NOME_DESCRITIVO.md` - Documentos temáticos (maiúsculo)

### Pasta setup/
- `snake_case.py` - Scripts Python
- `setup.py` - Configuração padrão
- `demo_*.py` - Demonstrações
- `test_*.py` - Scripts de teste
- `install_*.py` - Scripts de instalação

## 🎯 Objetivos da Organização

### Principais:
- **Eliminar redundâncias** e duplicações
- **Facilitar manutenção** e navegação
- **Padronizar** estrutura e nomenclatura
- **Otimizar** localização de informações

### Benefícios:
- **Projeto mais limpo** e profissional
- **Fácil localização** de arquivos
- **Manutenção simplificada**
- **Melhor experiência** para usuários/avaliadores

## 💡 Lembrete Importante

**Este arquivo serve como referência permanente** para manter a organização do projeto. Sempre que houver necessidade de reorganização ou limpeza, consulte estas diretrizes para manter a consistência e qualidade da estrutura.

## ✅ Status de Execução - Sistema Funcional

### Correções Realizadas (2025-01-24):
- **✅ Imports LangChain**: Atualizados para versões modernas (langchain-community, langchain-openai)
- **✅ Dependências**: Instaladas todas as dependências necessárias
- **✅ Demo funcionando**: `setup/demo_ecotravel.py` executa perfeitamente
- **✅ Teste de instalação**: `setup/test_installation.py` confirma sistema pronto
- **✅ Tratamento de APIs**: Verificação segura de chaves de API
- **✅ Imports seguros**: Try/except para múltiplas versões do LangChain

### Como Executar:
```bash
# Ativar ambiente virtual
source .venv/bin/activate

# Testar instalação
python setup/test_installation.py

# Executar demo (funciona sem API keys)
python setup/demo_ecotravel.py

# Para sistema completo, configure:
export OPENAI_API_KEY="sua-chave"
python src/agent/ecotravel_agent_rl.py
```

### Estrutura Final Organizada:
- **Projeto limpo** sem arquivos redundantes
- **Imports modernos** compatíveis com LangChain atual
- **Sistema executável** com fallbacks robustos
- **Documentação consolidada** e padronizada

---

**Última atualização**: Sistema reorganizado e corrigido para execução (2025-01-24)