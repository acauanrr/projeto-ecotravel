"""
Script para treinar o agente RL do EcoTravel Agent
Treina o agente para selecionar ferramentas otimamente
"""

import os
import sys
from pathlib import Path
import json
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Adicionar diretório pai ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rl.environment import EcoTravelEnvironment
from src.rl.rl_agent import RLAgent

# Configurações
TRAIN_EPISODES = 1000
EVAL_EPISODES = 100
SAVE_INTERVAL = 100
MODEL_DIR = Path("models")
METRICS_DIR = Path("metrics")
PLOTS_DIR = Path("plots")

# Criar diretórios
for dir_path in [MODEL_DIR, METRICS_DIR, PLOTS_DIR]:
    dir_path.mkdir(exist_ok=True)


class RLTrainer:
    """Classe para treinar e avaliar o agente RL"""
    
    def __init__(self):
        self.env = EcoTravelEnvironment()
        self.agent = RLAgent(env=self.env)
        self.training_metrics = {
            "episode_rewards": [],
            "episode_lengths": [],
            "tool_selections": [],
            "success_rates": [],
            "avg_latencies": []
        }
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def train(self, episodes: int = TRAIN_EPISODES):
        """Treina o agente RL"""
        print(f"Iniciando treinamento por {episodes} episódios...")
        
        for episode in range(episodes):
            # Reset ambiente
            state, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            tools_used = []
            successes = []
            latencies = []
            
            done = False
            while not done:
                # Agente escolhe ação
                action = self.agent.predict(state)
                
                # Executar ação
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Registrar métricas
                episode_reward += reward
                episode_length += 1
                tools_used.append(info["tool_used"])
                successes.append(info["success"])
                latencies.append(info["latency"])
                
                state = next_state
            
            # Salvar métricas do episódio
            self.training_metrics["episode_rewards"].append(episode_reward)
            self.training_metrics["episode_lengths"].append(episode_length)
            self.training_metrics["tool_selections"].append(tools_used)
            self.training_metrics["success_rates"].append(np.mean(successes))
            self.training_metrics["avg_latencies"].append(np.mean(latencies))
            
            # Treinar o agente com experiência acumulada
            if episode > 0 and episode % 10 == 0:
                self.agent.train(total_timesteps=1000)
            
            # Log de progresso
            if episode % 50 == 0:
                avg_reward = np.mean(self.training_metrics["episode_rewards"][-50:])
                avg_success = np.mean(self.training_metrics["success_rates"][-50:])
                print(f"Episódio {episode}: Recompensa média = {avg_reward:.2f}, "
                      f"Taxa de sucesso = {avg_success:.2%}")
            
            # Salvar modelo periodicamente
            if episode > 0 and episode % SAVE_INTERVAL == 0:
                self.save_model(f"checkpoint_ep{episode}")
        
        print("Treinamento concluído!")
        self.save_model("final")
        self.save_metrics()
    
    def evaluate(self, episodes: int = EVAL_EPISODES):
        """Avalia o desempenho do agente treinado"""
        print(f"\nAvaliando agente por {episodes} episódios...")
        
        eval_metrics = {
            "rewards": [],
            "tool_accuracy": [],
            "latencies": [],
            "costs": [],
            "query_types": [],
            "optimal_selections": []
        }
        
        for episode in range(episodes):
            state, _ = self.env.reset()
            query = self.env.current_query
            
            # Classificar tipo de query
            query_type = self._classify_query(query)
            eval_metrics["query_types"].append(query_type)
            
            done = False
            episode_info = {
                "tools": [],
                "successes": [],
                "latencies": [],
                "costs": []
            }
            
            while not done:
                # Predição do agente
                action = self.agent.predict(state, deterministic=True)
                
                # Executar
                state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Coletar informações
                episode_info["tools"].append(info["tool_used"])
                episode_info["successes"].append(info["success"])
                episode_info["latencies"].append(info["latency"])
                episode_info["costs"].append(info.get("cost", 0))
            
            # Calcular métricas
            eval_metrics["rewards"].append(reward)
            eval_metrics["latencies"].append(np.mean(episode_info["latencies"]))
            eval_metrics["costs"].append(sum(episode_info["costs"]))
            
            # Verificar se escolheu a ferramenta ótima
            optimal = self._check_optimal_selection(query, episode_info["tools"])
            eval_metrics["optimal_selections"].append(optimal)
            eval_metrics["tool_accuracy"].append(
                1.0 if optimal else 0.0
            )
        
        # Estatísticas finais
        results = {
            "avg_reward": np.mean(eval_metrics["rewards"]),
            "std_reward": np.std(eval_metrics["rewards"]),
            "tool_accuracy": np.mean(eval_metrics["tool_accuracy"]),
            "avg_latency": np.mean(eval_metrics["latencies"]),
            "avg_cost": np.mean(eval_metrics["costs"]),
            "performance_by_query_type": self._analyze_by_query_type(eval_metrics)
        }
        
        print("\n=== Resultados da Avaliação ===")
        print(f"Recompensa média: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"Precisão na seleção de ferramentas: {results['tool_accuracy']:.2%}")
        print(f"Latência média: {results['avg_latency']:.3f}s")
        print(f"Custo médio: ${results['avg_cost']:.4f}")
        
        print("\nDesempenho por tipo de query:")
        for query_type, metrics in results["performance_by_query_type"].items():
            print(f"  {query_type}: Precisão = {metrics['accuracy']:.2%}, "
                  f"Latência = {metrics['latency']:.3f}s")
        
        self.save_evaluation_results(results)
        self.create_visualizations(eval_metrics)
        
        return results
    
    def _classify_query(self, query: str) -> str:
        """Classifica o tipo de query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["clima", "tempo", "época"]):
            return "weather"
        elif any(word in query_lower for word in ["co2", "carbono", "emissão"]):
            return "carbon"
        elif any(word in query_lower for word in ["hotel", "destino", "eco"]):
            return "destinations"
        elif any(word in query_lower for word in ["atual", "recente", "novo"]):
            return "current_info"
        else:
            return "general"
    
    def _check_optimal_selection(self, query: str, tools_used: list) -> bool:
        """Verifica se a seleção de ferramentas foi ótima"""
        query_type = self._classify_query(query)
        
        optimal_tools = {
            "weather": ["Weather API"],
            "carbon": ["Carbon Calculator"],
            "destinations": ["RAG System"],
            "current_info": ["Web Search"],
            "general": ["RAG System", "Web Search"]
        }
        
        expected_tools = optimal_tools.get(query_type, [])
        
        # Verificar se usou pelo menos uma ferramenta esperada
        for tool in tools_used:
            if any(expected in tool for expected in expected_tools):
                return True
        
        return False
    
    def _analyze_by_query_type(self, metrics: dict) -> dict:
        """Analisa desempenho por tipo de query"""
        df = pd.DataFrame({
            "query_type": metrics["query_types"],
            "accuracy": metrics["tool_accuracy"],
            "latency": metrics["latencies"],
            "reward": metrics["rewards"]
        })
        
        results = {}
        for query_type in df["query_type"].unique():
            subset = df[df["query_type"] == query_type]
            results[query_type] = {
                "accuracy": subset["accuracy"].mean(),
                "latency": subset["latency"].mean(),
                "reward": subset["reward"].mean(),
                "count": len(subset)
            }
        
        return results
    
    def create_visualizations(self, eval_metrics: dict):
        """Cria visualizações dos resultados"""
        # 1. Evolução do treinamento
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        rewards_smooth = pd.Series(self.training_metrics["episode_rewards"]).rolling(50).mean()
        plt.plot(rewards_smooth)
        plt.title("Evolução da Recompensa")
        plt.xlabel("Episódio")
        plt.ylabel("Recompensa Média")
        
        plt.subplot(1, 3, 2)
        success_smooth = pd.Series(self.training_metrics["success_rates"]).rolling(50).mean()
        plt.plot(success_smooth)
        plt.title("Taxa de Sucesso")
        plt.xlabel("Episódio")
        plt.ylabel("Taxa de Sucesso")
        
        plt.subplot(1, 3, 3)
        latency_smooth = pd.Series(self.training_metrics["avg_latencies"]).rolling(50).mean()
        plt.plot(latency_smooth)
        plt.title("Latência Média")
        plt.xlabel("Episódio")
        plt.ylabel("Latência (s)")
        
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"training_evolution_{self.timestamp}.png")
        plt.close()
        
        # 2. Distribuição de ferramentas selecionadas
        plt.figure(figsize=(10, 6))
        all_tools = [tool for episode in self.training_metrics["tool_selections"] for tool in episode]
        tool_counts = pd.Series(all_tools).value_counts()
        
        sns.barplot(x=tool_counts.values, y=tool_counts.index)
        plt.title("Frequência de Uso de Ferramentas")
        plt.xlabel("Número de Usos")
        plt.ylabel("Ferramenta")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"tool_distribution_{self.timestamp}.png")
        plt.close()
        
        # 3. Performance por tipo de query
        query_types = eval_metrics["query_types"]
        accuracies = eval_metrics["tool_accuracy"]
        
        df = pd.DataFrame({"query_type": query_types, "accuracy": accuracies})
        plt.figure(figsize=(8, 6))
        
        sns.boxplot(data=df, x="query_type", y="accuracy")
        plt.title("Precisão por Tipo de Query")
        plt.xlabel("Tipo de Query")
        plt.ylabel("Precisão")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"accuracy_by_query_{self.timestamp}.png")
        plt.close()
        
        print(f"\nVisualizações salvas em {PLOTS_DIR}")
    
    def save_model(self, suffix: str = ""):
        """Salva o modelo treinado"""
        model_path = MODEL_DIR / f"rl_agent_{suffix}_{self.timestamp}.zip"
        self.agent.save(str(model_path))
        print(f"Modelo salvo em: {model_path}")
    
    def save_metrics(self):
        """Salva métricas de treinamento"""
        metrics_path = METRICS_DIR / f"training_metrics_{self.timestamp}.json"
        
        # Converter para formato serializável
        metrics_to_save = {
            "episode_rewards": self.training_metrics["episode_rewards"],
            "episode_lengths": self.training_metrics["episode_lengths"],
            "success_rates": self.training_metrics["success_rates"],
            "avg_latencies": self.training_metrics["avg_latencies"],
            "timestamp": self.timestamp,
            "total_episodes": len(self.training_metrics["episode_rewards"])
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics_to_save, f, indent=2)
        
        print(f"Métricas salvas em: {metrics_path}")
    
    def save_evaluation_results(self, results: dict):
        """Salva resultados da avaliação"""
        eval_path = METRICS_DIR / f"evaluation_results_{self.timestamp}.json"
        
        with open(eval_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Resultados da avaliação salvos em: {eval_path}")
    
    def compare_with_baseline(self):
        """Compara com baseline (seleção aleatória)"""
        print("\n=== Comparação com Baseline ===")
        
        # Baseline: seleção aleatória
        baseline_rewards = []
        baseline_accuracies = []
        
        for _ in range(EVAL_EPISODES):
            state, _ = self.env.reset()
            query = self.env.current_query
            
            # Ação aleatória
            action = self.env.action_space.sample()
            _, reward, _, _, info = self.env.step(action)
            
            baseline_rewards.append(reward)
            optimal = self._check_optimal_selection(query, [info["tool_used"]])
            baseline_accuracies.append(1.0 if optimal else 0.0)
        
        print(f"Baseline (aleatório):")
        print(f"  Recompensa média: {np.mean(baseline_rewards):.2f}")
        print(f"  Precisão: {np.mean(baseline_accuracies):.2%}")
        
        # Comparar com agente treinado
        eval_results = self.evaluate(episodes=EVAL_EPISODES)
        
        improvement = {
            "reward": (eval_results["avg_reward"] - np.mean(baseline_rewards)) / abs(np.mean(baseline_rewards)) * 100,
            "accuracy": (eval_results["tool_accuracy"] - np.mean(baseline_accuracies)) / np.mean(baseline_accuracies) * 100
        }
        
        print(f"\nMelhoria sobre baseline:")
        print(f"  Recompensa: +{improvement['reward']:.1f}%")
        print(f"  Precisão: +{improvement['accuracy']:.1f}%")


def main():
    """Função principal"""
    print("=== EcoTravel RL Agent - Treinamento ===\n")
    
    trainer = RLTrainer()
    
    # Treinar
    trainer.train(episodes=TRAIN_EPISODES)
    
    # Avaliar
    trainer.evaluate(episodes=EVAL_EPISODES)
    
    # Comparar com baseline
    trainer.compare_with_baseline()
    
    print("\n✅ Treinamento e avaliação concluídos!")


if __name__ == "__main__":
    main() 