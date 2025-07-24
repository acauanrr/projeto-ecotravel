import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import json
from datetime import datetime

# Importar nosso ambiente customizado
from src.rl.environment import EcoTravelRLEnvironment

class CustomPPONetwork(nn.Module):
    """
    Rede neural customizada para o PPO que processa embeddings de alta dimensão
    """
    def __init__(self, observation_space, action_space):
        super().__init__()
        
        n_input = observation_space.shape[0]
        n_output = action_space.n
        
        # Arquitetura profunda para processar embeddings complexos
        self.shared_net = nn.Sequential(
            nn.Linear(n_input, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Cabeças separadas para política e valor
        self.policy_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_output)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, obs):
        shared_features = self.shared_net(obs)
        return self.policy_head(shared_features), self.value_head(shared_features)

class EcoTravelRLAgent:
    """
    Agente RL para otimizar escolhas do EcoTravel Agent
    Usa PPO para aprender política de seleção de ferramentas
    """
    
    def __init__(self, 
                 model_name: str = "ecotravel_rl_v1",
                 use_advanced_embeddings: bool = True,
                 load_checkpoint: Optional[str] = None):
        
        self.model_name = model_name
        self.use_advanced_embeddings = use_advanced_embeddings
        
        # Criar ambiente
        self.env = Monitor(EcoTravelRLEnvironment(use_advanced_embeddings))
        self.vec_env = DummyVecEnv([lambda: self.env])
        
        # Configurar política customizada
        policy_kwargs = dict(
            features_extractor_class=CustomPPONetwork,
            features_extractor_kwargs=dict(
                observation_space=self.env.observation_space,
                action_space=self.env.action_space
            ),
            net_arch=[],  # Vazio pois usamos rede customizada
        )
        
        # Criar ou carregar modelo PPO
        if load_checkpoint:
            self.model = PPO.load(load_checkpoint, env=self.vec_env)
            print(f"Modelo carregado de {load_checkpoint}")
        else:
            self.model = PPO(
                "MlpPolicy",
                self.vec_env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.95,
                gae_lambda=0.95,
                clip_range=0.2,
                clip_range_vf=None,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                use_sde=True,  # Stochastic Differential Equations para melhor exploração
                sde_sample_freq=4,
                policy_kwargs=policy_kwargs,
                verbose=1,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
        
        # Configurar callbacks
        self.checkpoint_callback = CheckpointCallback(
            save_freq=1000,
            save_path=f"./models/{self.model_name}/checkpoints",
            name_prefix="rl_model"
        )
        
        self.eval_callback = EvalCallback(
            self.vec_env,
            best_model_save_path=f"./models/{self.model_name}/best",
            log_path=f"./models/{self.model_name}/logs",
            eval_freq=500,
            deterministic=True,
            render=False
        )
        
        # Métricas de treinamento
        self.training_metrics = {
            "episodes": 0,
            "total_reward": 0,
            "tool_usage": {0: 0, 1: 0, 2: 0, 3: 0},
            "success_rate": 0,
            "avg_latency": 0,
            "co2_saved": 0
        }
    
    def train(self, total_timesteps: int = 10000):
        """Treina o agente RL"""
        print(f"Iniciando treinamento por {total_timesteps} timesteps...")
        
        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=[self.checkpoint_callback, self.eval_callback],
                log_interval=10,
                progress_bar=True
            )
            
            # Salvar modelo final
            final_path = f"./models/{self.model_name}/final_model"
            self.model.save(final_path)
            print(f"Modelo final salvo em {final_path}")
            
        except KeyboardInterrupt:
            print("Treinamento interrompido. Salvando modelo atual...")
            self.model.save(f"./models/{self.model_name}/interrupted_model")
    
    def predict_tool(self, query: str, deterministic: bool = True) -> Dict[str, Any]:
        """
        Prevê qual ferramenta usar para uma query
        
        Args:
            query: Query do usuário
            deterministic: Se True, usa política determinística (melhor para produção)
        
        Returns:
            Dict com ferramenta recomendada e metadados
        """
        # Resetar ambiente com a query
        self.env.current_query = query
        obs = self.env.reset()[0]
        
        # Fazer predição
        action, _states = self.model.predict(obs, deterministic=deterministic)
        
        # Mapear ação para ferramenta
        tool_mapping = {
            0: "RAG",
            1: "API", 
            2: "Search",
            3: "Python"
        }
        
        # Calcular probabilidades de cada ferramenta
        if hasattr(self.model.policy, 'get_distribution'):
            dist = self.model.policy.get_distribution(obs.reshape(1, -1))
            probs = dist.distribution.probs.detach().cpu().numpy()[0]
        else:
            probs = None
        
        result = {
            "recommended_tool": tool_mapping[int(action)],
            "tool_index": int(action),
            "confidence": float(probs[action]) if probs is not None else 1.0,
            "all_probabilities": {
                tool_mapping[i]: float(probs[i]) if probs is not None else 0.0
                for i in range(4)
            },
            "query_features": {
                "length": len(query),
                "has_calculation": "calcul" in query.lower(),
                "has_weather": any(w in query.lower() for w in ["clima", "tempo"]),
                "has_sustainability": any(w in query.lower() for w in ["sustentável", "co2"])
            }
        }
        
        return result
    
    def online_learning(self, query: str, tool_used: int, 
                       success: bool, latency: float, user_feedback: float = 0.0):
        """
        Aprendizado online com feedback do usuário
        
        Args:
            query: Query processada
            tool_used: Índice da ferramenta usada
            success: Se a execução foi bem-sucedida
            latency: Latência real da execução
            user_feedback: Feedback do usuário (-1 a 1)
        """
        # Criar experiência customizada
        self.env.current_query = query
        obs = self.env.reset()[0]
        
        # Calcular recompensa ajustada com feedback
        base_reward = 2.0 if success else -1.0
        reward = base_reward + user_feedback
        
        # Armazenar experiência para replay buffer
        # (Isso seria mais complexo em produção, mas serve como exemplo)
        
        # Atualizar métricas
        self.training_metrics["episodes"] += 1
        self.training_metrics["total_reward"] += reward
        self.training_metrics["tool_usage"][tool_used] += 1
        
        # Treinar incrementalmente (mini-batch)
        if self.training_metrics["episodes"] % 10 == 0:
            self.model.learn(total_timesteps=100, reset_num_timesteps=False)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retorna métricas de desempenho do agente"""
        total_uses = sum(self.training_metrics["tool_usage"].values())
        
        if total_uses > 0:
            tool_distribution = {
                f"tool_{k}": v / total_uses 
                for k, v in self.training_metrics["tool_usage"].items()
            }
        else:
            tool_distribution = {}
        
        return {
            "episodes_trained": self.training_metrics["episodes"],
            "average_reward": self.training_metrics["total_reward"] / max(1, self.training_metrics["episodes"]),
            "tool_distribution": tool_distribution,
            "model_info": {
                "name": self.model_name,
                "use_advanced_embeddings": self.use_advanced_embeddings,
                "device": str(self.model.device)
            }
        }
    
    def save_metrics(self, filepath: str = None):
        """Salva métricas em arquivo JSON"""
        if filepath is None:
            filepath = f"./models/{self.model_name}/metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        metrics = self.get_metrics()
        metrics["timestamp"] = datetime.now().isoformat()
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Métricas salvas em {filepath}")


# Exemplo de uso
if __name__ == "__main__":
    # Criar e treinar agente
    agent = EcoTravelRLAgent(use_advanced_embeddings=True)
    
    # Treinar (reduzido para demonstração)
    agent.train(total_timesteps=5000)
    
    # Testar predições
    test_queries = [
        "Calcule as emissões de CO2 de SP para RJ",
        "Qual o clima no Rio amanhã?",
        "Encontre hotéis sustentáveis em Florianópolis",
        "Compare rotas ecológicas entre cidades"
    ]
    
    for query in test_queries:
        result = agent.predict_tool(query)
        print(f"\nQuery: {query}")
        print(f"Ferramenta recomendada: {result['recommended_tool']}")
        print(f"Confiança: {result['confidence']:.2%}")
        print(f"Probabilidades: {result['all_probabilities']}")
    
    # Salvar métricas
    agent.save_metrics() 