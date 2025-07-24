import gymnasium as gym
import numpy as np
from gymnasium import spaces
import os
from typing import Dict, Tuple, Any, List
import openai
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
import time

# Configurar APIs com segurança
openai.api_key = os.getenv('OPENAI_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
HF_TOKEN = os.getenv('HF_TOKEN')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')

@dataclass
class ToolMetrics:
    """Métricas para cada ferramenta"""
    name: str
    avg_latency: float
    success_rate: float
    cost_per_use: float
    co2_impact: float

class EcoTravelRLEnvironment(gym.Env):
    """
    Ambiente RL para otimizar escolhas do EcoTravel Agent
    
    Estados: Embeddings da query + contexto + métricas históricas
    Ações: Escolha de ferramenta (RAG, API, Search, Python)
    Recompensas: Multi-objetivo (precisão, latência, CO2, custo)
    """
    
    def __init__(self, use_advanced_embeddings: bool = True):
        super().__init__()
        
        # Configurar encoder de embeddings
        if use_advanced_embeddings:
            # Usar OpenAI embeddings para melhor qualidade
            self.embedding_model = "openai"
        else:
            # Fallback para modelo local
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Definir espaços
        self.embedding_dim = 1536 if use_advanced_embeddings else 384
        self.n_tools = 4  # RAG, API, Search, Python
        self.context_features = 10  # Features adicionais de contexto
        
        # Estado: embedding da query + features de contexto + métricas históricas
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.embedding_dim + self.context_features + self.n_tools * 4,),
            dtype=np.float32
        )
        
        # Ação: escolha de ferramenta
        self.action_space = spaces.Discrete(self.n_tools)
        
        # Métricas das ferramentas
        self.tool_metrics = {
            0: ToolMetrics("RAG", avg_latency=0.8, success_rate=0.85, cost_per_use=0.001, co2_impact=0.1),
            1: ToolMetrics("API", avg_latency=1.2, success_rate=0.95, cost_per_use=0.002, co2_impact=0.2),
            2: ToolMetrics("Search", avg_latency=2.0, success_rate=0.75, cost_per_use=0.003, co2_impact=0.3),
            3: ToolMetrics("Python", avg_latency=0.5, success_rate=0.99, cost_per_use=0.0005, co2_impact=0.05)
        }
        
        # Estado inicial
        self.reset()
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Gera embedding usando OpenAI ou modelo local"""
        if self.embedding_model == "openai":
            try:
                response = openai.Embedding.create(
                    input=text,
                    model="text-embedding-3-large"
                )
                return np.array(response['data'][0]['embedding'])
            except Exception as e:
                print(f"Erro com OpenAI embeddings: {e}. Usando modelo local.")
                # Fallback para modelo local
                model = SentenceTransformer('all-MiniLM-L6-v2')
                return model.encode(text)
        else:
            return self.embedding_model.encode(text)
    
    def _extract_context_features(self, query: str) -> np.ndarray:
        """Extrai features de contexto da query"""
        features = []
        
        # Feature 1-2: Comprimento da query (normalizado)
        features.append(len(query) / 500.0)
        features.append(len(query.split()) / 50.0)
        
        # Feature 3-5: Presença de palavras-chave
        features.append(1.0 if "sustentável" in query.lower() else 0.0)
        features.append(1.0 if "co2" in query.lower() or "carbono" in query.lower() else 0.0)
        features.append(1.0 if "custo" in query.lower() or "preço" in query.lower() else 0.0)
        
        # Feature 6-7: Complexidade estimada
        features.append(1.0 if "?" in query else 0.0)
        features.append(query.count(",") / 10.0)
        
        # Feature 8-10: Tipo de requisição
        features.append(1.0 if "calcul" in query.lower() else 0.0)
        features.append(1.0 if "clima" in query.lower() or "tempo" in query.lower() else 0.0)
        features.append(1.0 if "hotel" in query.lower() or "hospedagem" in query.lower() else 0.0)
        
        return np.array(features)
    
    def _get_tool_history_features(self) -> np.ndarray:
        """Retorna features históricas de desempenho das ferramentas"""
        features = []
        for i in range(self.n_tools):
            metrics = self.tool_metrics[i]
            features.extend([
                metrics.avg_latency / 3.0,  # Normalizado
                metrics.success_rate,
                metrics.cost_per_use * 1000,  # Escalonado
                metrics.co2_impact
            ])
        return np.array(features)
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reseta o ambiente com uma nova query"""
        super().reset(seed=seed)
        
        # Simular queries de exemplo
        sample_queries = [
            "Quero viajar de São Paulo para o Rio de forma sustentável",
            "Calcule as emissões de CO2 de uma viagem de avião SP-RJ",
            "Qual a previsão do tempo no Rio para próxima semana?",
            "Encontre hotéis eco-friendly no Rio com bom custo-benefício",
            "Compare rotas sustentáveis entre SP e RJ considerando tempo e CO2"
        ]
        
        self.current_query = np.random.choice(sample_queries)
        
        # Gerar estado inicial
        query_embedding = self._get_embedding(self.current_query)
        context_features = self._extract_context_features(self.current_query)
        tool_features = self._get_tool_history_features()
        
        # Ajustar dimensão se necessário
        if len(query_embedding) < self.embedding_dim:
            query_embedding = np.pad(query_embedding, (0, self.embedding_dim - len(query_embedding)))
        
        self.state = np.concatenate([query_embedding, context_features, tool_features])
        
        return self.state.astype(np.float32), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Executa ação e calcula recompensa"""
        assert self.action_space.contains(action)
        
        tool = self.tool_metrics[action]
        
        # Simular execução da ferramenta
        start_time = time.time()
        success = np.random.random() < tool.success_rate
        actual_latency = tool.avg_latency + np.random.normal(0, 0.2)
        
        # Calcular recompensa multi-objetivo
        reward = self._calculate_reward(
            tool=tool,
            success=success,
            latency=actual_latency,
            query=self.current_query,
            action=action
        )
        
        # Atualizar métricas históricas (simulado)
        self.tool_metrics[action].avg_latency = 0.9 * tool.avg_latency + 0.1 * actual_latency
        
        # Episode termina após uma escolha
        terminated = True
        truncated = False
        
        # Informações adicionais
        info = {
            "tool_used": tool.name,
            "success": success,
            "latency": actual_latency,
            "co2_saved": 1.0 - tool.co2_impact if success else 0,
            "query": self.current_query
        }
        
        return self.state, reward, terminated, truncated, info
    
    def _calculate_reward(self, tool: ToolMetrics, success: bool, 
                         latency: float, query: str, action: int) -> float:
        """
        Calcula recompensa multi-objetivo considerando:
        - Sucesso da execução
        - Latência
        - Impacto de CO2
        - Custo
        - Adequação da ferramenta para a query
        """
        reward = 0.0
        
        # Recompensa base por sucesso
        if success:
            reward += 2.0
        else:
            reward -= 1.0
        
        # Penalidade por latência (quanto menor, melhor)
        latency_penalty = -0.5 * (latency / 3.0)  # Normalizado para max 3s
        reward += latency_penalty
        
        # Bonus por baixo impacto de CO2
        co2_bonus = 0.5 * (1.0 - tool.co2_impact)
        reward += co2_bonus
        
        # Penalidade por custo
        cost_penalty = -0.3 * (tool.cost_per_use / 0.005)  # Normalizado
        reward += cost_penalty
        
        # Bonus por adequação da ferramenta à query
        adequacy_bonus = self._calculate_tool_adequacy(query, action)
        reward += adequacy_bonus
        
        return reward
    
    def _calculate_tool_adequacy(self, query: str, action: int) -> float:
        """Calcula quão adequada é a ferramenta para a query"""
        query_lower = query.lower()
        
        # RAG (action 0): bom para informações sobre destinos, guias
        if action == 0:
            if any(word in query_lower for word in ["hotel", "guia", "destino", "eco-friendly"]):
                return 1.0
            return 0.0
        
        # API (action 1): bom para dados em tempo real
        elif action == 1:
            if any(word in query_lower for word in ["clima", "tempo", "previsão"]):
                return 1.0
            return 0.0
        
        # Search (action 2): bom para informações atuais, eventos
        elif action == 2:
            if any(word in query_lower for word in ["evento", "notícia", "atual"]):
                return 1.0
            return 0.0
        
        # Python (action 3): bom para cálculos
        elif action == 3:
            if any(word in query_lower for word in ["calcul", "co2", "emissão", "compar"]):
                return 1.0
            return 0.0
        
        return 0.0 