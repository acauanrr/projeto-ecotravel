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
    
    def __init__(self, use_advanced_embeddings: bool = False):
        super().__init__()
        
        # Configurar encoder de embeddings
        # Por padrão, usar modelo local para evitar dependência de API
        if use_advanced_embeddings and os.getenv('OPENAI_API_KEY'):
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
                # Usar API nova do OpenAI
                client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
                response = client.embeddings.create(
                    input=text,
                    model="text-embedding-3-small"  # Modelo mais leve
                )
                return np.array(response.data[0].embedding)
            except Exception as e:
                print(f"Erro com OpenAI embeddings: {e}. Usando modelo local.")
                # Fallback para modelo local
                if not hasattr(self, '_local_model'):
                    self._local_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.embedding_model = self._local_model
                return self._local_model.encode(text)
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
    
    def reset(self, seed=None, options=None):
        """Reset do ambiente"""
        super().reset(seed=seed)
        
        # Usar queries reais baseadas em dados do Brasil
        real_queries = [
            # Queries sobre destinos sustentáveis
            "Quais são os melhores destinos sustentáveis no Nordeste do Brasil?",
            "Como visitar Fernando de Noronha de forma ecológica?",
            "Onde encontrar hotéis eco-friendly em Gramado?",
            "Quais praias do Brasil têm certificação ambiental?",
            
            # Queries sobre transporte
            "Como ir de São Paulo para Rio de forma sustentável?",
            "Qual é a pegada de carbono de um voo SP-Salvador?",
            "Vale a pena ir de ônibus de Curitiba para Florianópolis?",
            "Como calcular emissões de CO2 de uma viagem de carro?",
            
            # Queries sobre clima e época
            "Qual a melhor época para visitar a Amazônia?",
            "Como está o clima em Bonito no verão?",
            "Quando é a temporada de chuvas no Pantanal?",
            
            # Queries sobre informações gerais
            "O que é turismo regenerativo?",
            "Como compensar emissões de carbono de viagens?",
            "Quais são as certificações ambientais para hotéis?",
            
            # Queries complexas
            "Planeje uma viagem sustentável de 7 dias pelo Nordeste",
            "Compare emissões de diferentes transportes para ir de Brasília a Goiânia",
            "Quais atividades eco-friendly fazer em Foz do Iguaçu?"
        ]
        
        # Selecionar query real aleatória
        self.current_query = self.np_random.choice(real_queries)
        
        # Gerar embedding real da query usando um modelo simples
        # Em produção, usar sentence-transformers ou similar
        query_length = len(self.current_query.split())
        query_complexity = sum(1 for word in self.current_query.split() if len(word) > 5)
        
        # Features mais realistas baseadas na query
        query_features = np.array([
            query_length / 20.0,  # Normalizado
            query_complexity / 10.0,
            1.0 if "sustentável" in self.current_query.lower() else 0.0,
            1.0 if "carbono" in self.current_query.lower() or "co2" in self.current_query.lower() else 0.0,
            1.0 if "clima" in self.current_query.lower() else 0.0,
            1.0 if "hotel" in self.current_query.lower() or "hospedagem" in self.current_query.lower() else 0.0,
            1.0 if any(city in self.current_query for city in ["São Paulo", "Rio", "Salvador", "Nordeste"]) else 0.0,
            1.0 if "?" in self.current_query else 0.0
        ])
        
        # Preencher resto do estado com valores realistas
        remaining_dims = self.observation_space.shape[0] - len(query_features)
        padding = np.zeros(remaining_dims)
        
        self.state = np.concatenate([query_features, padding]).astype(np.float32)
        
        # Reset contadores
        self.step_count = 0
        self.tools_used = []
        self.total_cost = 0.0
        self.total_latency = 0.0
        
        return self.state, {}
    
    def step(self, action):
        """Executa ação no ambiente"""
        # Validar action
        if not isinstance(action, (int, np.integer)):
            raise ValueError(f"Action deve ser um inteiro, recebido: {type(action)}")
        
        if action not in self.tool_metrics:
            raise ValueError(f"Action {action} inválida. Deve estar entre 0 e {self.n_tools-1}")
        
        tool = self.tool_metrics[action]
        self.tools_used.append(tool.name)
        self.step_count += 1
        
        # Calcular sucesso baseado em heurísticas reais
        success_probability = self._calculate_real_success_probability(tool.name, self.current_query)
        success = self.np_random.random() < success_probability
        
        # Latência real baseada no tipo de ferramenta
        base_latency = tool.avg_latency
        if tool.name == "RAG":
            # RAG pode ser mais lento com queries complexas
            query_complexity = len(self.current_query.split()) / 10
            actual_latency = base_latency * (1 + query_complexity * 0.2)
        elif tool.name == "API":
            # APIs externas têm latência variável
            actual_latency = base_latency + self.np_random.normal(0, 0.5)
        else:
            actual_latency = base_latency + self.np_random.normal(0, 0.1)
        
        actual_latency = max(0.1, actual_latency)  # Mínimo 100ms
        
        # Acumular custos e latência
        self.total_cost += tool.cost_per_use
        self.total_latency += actual_latency
        
        # Calcular recompensa baseada em critérios reais
        reward = self._calculate_real_reward(tool.name, success, actual_latency)
        
        # Atualizar estado com informações da execução
        self.state = self._update_state_after_tool_use(tool.name, success)
        
        # Terminar se usado muitas ferramentas ou se encontrou resposta completa
        terminated = (self.step_count >= self.n_tools or 
                     (success and self._is_query_answered()))
        
        truncated = False
        
        try:
            info = {
                "tool_used": tool.name if hasattr(tool, 'name') else f"Tool_{action}",
                "success": success,
                "latency": actual_latency,
                "total_cost": self.total_cost,
                "query": self.current_query,
                "tools_sequence": self.tools_used
            }
        except Exception as e:
            print(f"Erro ao criar info: {e}")
            print(f"tool type: {type(tool)}, action: {action}")
            raise
        
        return self.state, reward, terminated, truncated, info
    
    def _calculate_real_success_probability(self, tool_name: str, query: str) -> float:
        """Calcula probabilidade de sucesso baseada em heurísticas reais"""
        query_lower = query.lower()
        
        # RAG - alta taxa de sucesso para queries sobre destinos e informações gerais
        if tool_name == "RAG":
            if any(word in query_lower for word in ["destino", "hotel", "sustentável", "eco", "certificação"]):
                return 0.85
            return 0.7
        
        # API - perfeita para queries sobre clima
        elif tool_name == "API":
            if any(word in query_lower for word in ["clima", "tempo", "chuva", "temperatura", "época"]):
                return 0.95
            return 0.3
        
        # Search - boa para informações atuais e específicas
        elif tool_name == "Search":
            if any(word in query_lower for word in ["atual", "2024", "recente", "novo"]):
                return 0.8
            return 0.6
        
        # Python - específica para cálculos
        elif tool_name == "Python":
            if any(word in query_lower for word in ["carbono", "co2", "emissão", "emissões", "pegada", "calcular emissões"]):
                return 0.9
            return 0.2
        
        return 0.5  # Default
    
    def _calculate_real_reward(self, tool_name: str, success: bool, latency: float) -> float:
        """Calcula recompensa realista baseada no resultado"""
        if not success:
            return -3.0  # Penalidade por falha
        
        # Recompensa base por sucesso
        reward = 10.0
        
        # Bônus por escolher a ferramenta certa para a query
        if self._is_optimal_tool_for_query(tool_name, self.current_query):
            reward += 5.0
        
        # Penalidade por latência (suave)
        latency_penalty = min(latency * 0.5, 3.0)
        reward -= latency_penalty
        
        # Penalidade por uso excessivo de ferramentas
        if len(self.tools_used) > 2:
            reward -= (len(self.tools_used) - 2) * 1.0
        
        # Penalidade por repetir ferramenta desnecessariamente
        if self.tools_used.count(tool_name) > 1:
            reward -= 2.0
        
        return reward
    
    def _is_optimal_tool_for_query(self, tool_name: str, query: str) -> bool:
        """Verifica se é a ferramenta ideal para a query"""
        query_lower = query.lower()
        
        optimal_mappings = {
            "RAG": ["destino", "hotel", "sustentável", "eco", "certificação", "turismo"],
            "API": ["clima", "tempo", "chuva", "temperatura", "época", "temporada"],
            "Search": ["atual", "2024", "recente", "novo", "informação"],
            "Python": ["carbono", "co2", "emissão", "emissões", "pegada", "calcular emissões"]
        }
        
        tool_keywords = optimal_mappings.get(tool_name, [])
        return any(keyword in query_lower for keyword in tool_keywords)
    
    def _is_query_answered(self) -> bool:
        """Verifica se a query foi adequadamente respondida"""
        # Heurística: se usou a ferramenta certa com sucesso
        for tool_name in self.tools_used:
            if self._is_optimal_tool_for_query(tool_name, self.current_query):
                return True
        
        # Ou se usou múltiplas ferramentas complementares
        return len(set(self.tools_used)) >= 2
    
    def _update_state_after_tool_use(self, tool_name: str, success: bool) -> np.ndarray:
        """Atualiza o estado após usar uma ferramenta"""
        new_state = self.state.copy()
        
        # Atualizar features baseadas no uso da ferramenta
        tool_index = next(key for key, tool in self.tool_metrics.items() if tool.name == tool_name)
        
        # Marcar ferramenta como usada
        if tool_index < 10:  # Limitar para não ultrapassar dimensões
            new_state[8 + tool_index] = 1.0
        
        # Indicar sucesso/falha
        new_state[18] = 1.0 if success else -1.0
        
        # Atualizar contador normalizado
        new_state[19] = min(self.step_count / self.n_tools, 1.0)
        
        return new_state.astype(np.float32) 