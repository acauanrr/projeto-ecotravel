# EcoTravel Agent Dockerfile
FROM python:3.9-slim

# Definir diretório de trabalho
WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primeiro para cache de layers
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fonte
COPY src/ ./src/
COPY data/ ./data/
COPY notebooks/ ./notebooks/
COPY docs/ ./docs/
COPY .env.example .env

# Criar usuário não-root
RUN useradd -m -u 1000 ecotravel && chown -R ecotravel:ecotravel /app
USER ecotravel

# Expor porta para aplicações web
EXPOSE 8000

# Comando padrão
CMD ["python", "-c", "from src.agent.eco_travel_agent import EcoTravelAgent; EcoTravelAgent().chat()"]