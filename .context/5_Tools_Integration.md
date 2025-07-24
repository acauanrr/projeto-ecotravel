# Integração de Ferramentas

## Ferramentas Adicionais
- **API Open-Meteo**:
  - Função: Previsão do tempo e alertas climáticos.
  - Uso: Consultas HTTP via `requests` (ex.: clima em destino).
- **DuckDuckGo Search**:
  - Função: Busca de eventos e notícias locais.
  - Uso: LangChain tool (`DuckDuckGoSearchRun`).
- **Interpretador Python**:
  - Função: Cálculos de pegada de carbono, otimização de rotas, análise de custos.
  - Uso: LangChain `PythonREPLTool` ou função custom.

## Exemplo: Calculadora de Carbono
```python
from langchain.tools import Tool

def calculate_carbon_footprint(transport_mode, distance):
    """Calcula pegada de carbono em kg CO2."""
    emissions = {
        "aviao": 0.255,  # kg CO2/km
        "carro": 0.171,
        "trem": 0.041,
        "onibus": 0.089
    }
    return emissions.get(transport_mode, 0) * distance

carbon_tool = Tool(
    name="CarbonCalculator",
    func=calculate_carbon_footprint,
    description="Calcula emissões de CO2 para diferentes transportes"
)
```

## Configuração
- **Open-Meteo**: Use `requests.get("https://api.open-meteo.com/v1/forecast?...")`.
- **DuckDuckGo**: Integre via `langchain.tools.DuckDuckGoSearchRun`.
- **Python**: Use `PythonREPLTool` para cálculos dinâmicos.

## Integração com Agente
```python
from langchain.tools import DuckDuckGoSearchRun, PythonREPLTool

tools = [
    carbon_tool,
    DuckDuckGoSearchRun(name="WebSearch"),
    PythonREPLTool(name="PythonCalc"),
    # RAG tool será adicionado separadamente
]
```

## Validação
- **Testes**: Simule queries como "calcular CO2 de trem SP->RJ" ou "eventos no RJ hoje".
- **Métricas**: Tempo de resposta, precisão dos cálculos (comparar com valores esperados).