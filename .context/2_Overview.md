# EcoTravel Agent - Visão Geral

## Objetivo
Desenvolver um assistente inteligente para planejamento de viagens sustentáveis, integrando Modelos de Linguagem (LLMs) com agentes autônomos, Recuperação de Informação (RAG) e ferramentas externas, atendendo aos requisitos do TP 5 - Agentes com LLMs.

## Problema
Criar um agente que planeje viagens sustentáveis com:
- **Análise de Pegada de Carbono**: Calcula emissões por modal de transporte.
- **Orçamento Inteligente**: Otimiza custos com alternativas ecológicas.
- **Recomendações Culturais**: Sugere experiências locais sustentáveis.
- **Alertas em Tempo Real**: Monitora clima e eventos locais.

## Justificativa
- **Complexidade Multi-dimensional**: Requer orquestração de RAG (guias locais) e ferramentas externas (APIs, cálculos).
- **Valor Prático**: Reduz emissões de CO2, economiza custos e melhora a experiência do usuário.
- **Métricas Claras**: Avaliação via redução de CO2, economia financeira e precisão das recomendações.
- **Viabilidade**: Usa ferramentas gratuitas (Ollama, LangChain, APIs públicas) e roda localmente ou no Google Colab.

## Pontos Fortes
- Atende ao requisito de RAG (base de conhecimento com PDFs/CSVs).
- Integra ferramentas adicionais (API Open-Meteo, DuckDuckGo Search, Python).
- Resolve um problema real (sustentabilidade em viagens).
- Implementável com recursos gratuitos.
- Demonstra impacto mensurável (ex.: economia de tempo e CO2).

## Diferenciais
- Benchmarks quantitativos (ex.: Hit Rate, MRR para RAG).
- Interface interativa (Gradio/Streamlit).
- Testes extensivos para edge cases.
- Documentação visual profissional (fluxogramas).
- Análise de impacto real (ex.: redução de CO2 quantificada).