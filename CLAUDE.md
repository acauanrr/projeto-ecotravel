# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EcoTravel Agent is an intelligent sustainable travel planning system that combines Reinforcement Learning (RL) with RAG (Retrieval-Augmented Generation) and multi-tool capabilities. The system provides eco-friendly travel recommendations by integrating multiple AI technologies and APIs.

## Development Commands

### Environment Setup
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Alternative robust installation
python setup/install_dependencies.py
```

### Testing and Validation
```bash
# Complete system test (always works)
python teste_completo.py

# Interactive demo (works without APIs)
python setup/demo_ecotravel.py

# Installation verification
python setup/test_installation.py

# Run unit tests
python -m pytest src/tests/

# System benchmark
python src/tests/test_system.py --benchmark

# Full test report
python src/tests/test_system.py --report
```

### Development Tools
```bash
# Code formatting
black src/

# Linting
pylint src/

# Dashboard (interactive metrics)
streamlit run src/dashboard/metrics_dashboard.py

# Jupyter notebooks
jupyter notebook notebooks/
```

### Package Management
```bash
# Install as editable package
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"

# Install all extras
pip install -e ".[all]"
```

## Architecture Overview

The system follows a modular architecture with these core components:

### 1. Agent Layer (`src/agent/`)
- **`eco_travel_agent.py`**: Basic agent without RL
- **`ecotravel_agent_rl.py`**: Advanced agent with RL integration using LangChain ReAct pattern

### 2. Reinforcement Learning (`src/rl/`)
- **`environment.py`**: Custom Gymnasium environment for tool selection optimization
- **`rl_agent.py`**: PPO agent using Stable-Baselines3 for learning optimal tool selection policies

### 3. RAG System (`src/rag/`)
- **`rag_system.py`**: Advanced RAG with hybrid search (BM25 + semantic), reranking, and anti-hallucination features
- Uses RecursiveCharacterTextSplitter for intelligent chunking (512 chars, 50 overlap)
- FAISS vector store with sentence-transformers embeddings

### 4. Tools (`src/tools/`)
- **`carbon_calculator.py`**: CO2 emissions calculation for different transport modes
- **`weather_api.py`**: Real-time weather data via Open-Meteo API
- **`web_search.py`**: Web search using DuckDuckGo

### 5. Data Layer (`data/`)
- **`guias/`**: Sustainable travel guides for Brazil
- **`emissoes/`**: Transport emissions datasets (CSV)
- **`avaliacoes/`**: Sustainable hotels database (JSON)

## Key Configuration

### Environment Variables
Create a `.env` file with:
```bash
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_google_key
HF_TOKEN=your_huggingface_token  # Optional
DEEPSEEK_API_KEY=your_deepseek_key  # Optional
```

### API Dependencies
- **OpenAI**: GPT-4 + text-embedding-3 (primary LLM and embeddings)
- **Open-Meteo**: Free weather API (no key required)
- **DuckDuckGo**: Free search API (no key required)
- **Google Search**: Optional for enhanced search results

## Development Patterns

### Agent Integration
The RL agent optimizes tool selection using a multi-objective reward function:
- Precision: Accuracy of tool recommendations
- Latency: Response time optimization
- Cost: API usage minimization
- CO2 Impact: Environmental consideration

### RAG Strategy
- **Hybrid Search**: Combines lexical (BM25) and semantic search
- **Reranking**: Cross-encoder for improved result ordering
- **Source Verification**: Anti-hallucination through source citation

### Error Handling
The system includes robust fallbacks:
- LangChain import compatibility across versions
- Local embedding models as OpenAI fallbacks
- Offline functionality for core features

## Performance Characteristics

### Benchmarks
- **RAG Response Time**: <500ms average
- **RL Prediction Time**: <100ms
- **Complete Agent Response**: 1.6s average (35% improvement over baseline)
- **Memory Usage**: ~500MB-1GB typical

### Quality Metrics
- **RAG Hit Rate**: >80%
- **RL Convergence**: 85% in 5k episodes
- **Tool Selection Accuracy**: 92%

## Common Workflows

### Adding New Tools
1. Create tool function in `src/tools/`
2. Register as LangChain Tool with proper description
3. Add to agent's tool list in `ecotravel_agent_rl.py`
4. Update RL environment action space if needed

### Extending Knowledge Base
1. Add documents to appropriate `data/` subdirectory
2. Run `rag_system.build_index()` to rebuild vector store
3. Verify new content with test queries

### RL Training
```python
from src.rl.rl_agent import EcoTravelRLAgent

rl_agent = EcoTravelRLAgent()
rl_agent.train(total_timesteps=5000)
rl_agent.save("models/trained_agent")
```

## Testing Strategy

### Functional Tests
- **Demo Mode**: `python setup/demo_ecotravel.py` (always works, no APIs required)
- **API Tests**: Validates external API connectivity
- **Component Tests**: Individual tool and system testing

### Integration Tests
- **Complete System**: `python teste_completo.py`
- **Performance Benchmarks**: Measure response times and accuracy
- **RL Training Verification**: Convergence and policy quality tests

## Troubleshooting

### Common Issues
1. **Import Errors**: LangChain version compatibility handled with try/except blocks
2. **API Failures**: Graceful degradation with offline alternatives
3. **Memory Issues**: Large embedding models - consider using smaller alternatives
4. **RL Training**: Simulated environment available if full RL training fails

### Debug Mode
Enable verbose logging:
```python
agent = EcoTravelAgentWithRL(verbose=True, debug=True)
```

## Entry Points
The package provides console commands:
- `ecotravel`: Run basic agent
- `ecotravel-test`: Execute test suite
- `ecotravel-benchmark`: Performance benchmarking