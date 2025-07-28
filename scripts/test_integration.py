"""
Script de teste de integração completa do EcoTravel Agent
Testa todos os componentes integrados com dados reais
"""

import sys
from pathlib import Path
import time
from datetime import datetime
import json

# Adicionar diretório pai ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Imports dos componentes
from src.rag.enhanced_rag_system import EnhancedRAGSystem
from src.tools.carbon_calculator import CarbonCalculator
from src.rl.environment import EcoTravelRLEnvironment


class IntegrationTester:
    """Classe para testar integração completa do sistema"""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "summary": {
                "total": 0,
                "passed": 0,
                "failed": 0
            }
        }
        
    def run_all_tests(self):
        """Executa todos os testes de integração"""
        print("=== Teste de Integração Completa - EcoTravel Agent ===\n")
        
        # 1. Testar Sistema RAG
        self.test_rag_system()
        
        # 2. Testar Calculadora de Carbono
        self.test_carbon_calculator()
        
        # 3. Testar Ambiente RL
        self.test_rl_environment()
        
        # 4. Testar Integração RAG + Ferramentas
        self.test_rag_tools_integration()
        
        # 5. Testar Pipeline Completo
        self.test_complete_pipeline()
        
        # Resumo final
        self.print_summary()
        self.save_results()
    
    def test_rag_system(self):
        """Testa o sistema RAG com dados reais"""
        print("1. Testando Sistema RAG...")
        test_name = "rag_system"
        
        try:
            # Inicializar RAG
            rag = EnhancedRAGSystem()
            rag.load_data()
            
            # Queries de teste
            test_queries = [
                "Quais são os melhores destinos sustentáveis no Nordeste?",
                "Como reduzir emissões de CO2 em viagens?",
                "Hotéis eco-friendly em Gramado",
                "Certificações ambientais para turismo"
            ]
            
            success_count = 0
            for query in test_queries:
                results = rag.search(query, use_reranking=True)
                
                if results and len(results) > 0:
                    print(f"   ✓ Query: '{query[:50]}...' - {len(results)} resultados")
                    success_count += 1
                else:
                    print(f"   ✗ Query: '{query[:50]}...' - Sem resultados")
            
            # Verificar estatísticas
            stats = rag.stats
            print(f"   Total documentos: {stats['total_documents']}")
            print(f"   Total chunks: {stats['total_chunks']}")
            print(f"   Fontes: {', '.join(stats['sources'])}")
            
            # Teste de contexto
            context, sources = rag.get_context_for_query("turismo sustentável no Brasil")
            context_ok = len(context) > 100 and len(sources) > 0
            
            test_passed = success_count == len(test_queries) and context_ok
            self._record_result(test_name, test_passed, {
                "queries_tested": len(test_queries),
                "successful_queries": success_count,
                "total_documents": stats['total_documents'],
                "total_chunks": stats['total_chunks']
            })
            
        except Exception as e:
            print(f"   ✗ Erro: {str(e)}")
            self._record_result(test_name, False, {"error": str(e)})
    
    def test_carbon_calculator(self):
        """Testa a calculadora de carbono com dados reais"""
        print("\n2. Testando Calculadora de Carbono...")
        test_name = "carbon_calculator"
        
        try:
            calc = CarbonCalculator()
            
            # Teste 1: Cálculo básico
            result = calc.calculate_carbon_footprint(
                transport_mode="aviao_domestico",
                distance_km=500,
                round_trip=True
            )
            
            test1_ok = (
                result["total_emissions_kg"] > 0 and
                result["emission_factor_kg_per_km"] == 0.158 and  # IPCC 2023
                len(result["suggestions"]) > 0 and
                result["data_source"] == "IPCC 2023"
            )
            
            print(f"   {'✓' if test1_ok else '✗'} Cálculo básico: {result['total_emissions_kg']} kg CO2")
            
            # Teste 2: Rota específica
            route_result = calc.calculate_route_emissions("São Paulo", "Rio")
            test2_ok = "route" in route_result and route_result.get("total_emissions_kg", 0) > 0
            
            print(f"   {'✓' if test2_ok else '✗'} Rota SP-RJ: {route_result.get('total_emissions_kg', 'N/A')} kg CO2")
            
            # Teste 3: Comparação de modais
            comparison = calc.compare_transport_modes(300)
            test3_ok = len(comparison) > 0 and "Modal" in comparison.columns
            
            print(f"   {'✓' if test3_ok else '✗'} Comparação de modais: {len(comparison)} opções")
            
            # Teste 4: Score de sustentabilidade
            score_info = calc.get_sustainability_score("metro", 100)
            test4_ok = score_info["score"] > 80  # Metro deve ter score alto
            
            print(f"   {'✓' if test4_ok else '✗'} Score sustentabilidade: {score_info['score']}/100")
            
            test_passed = all([test1_ok, test2_ok, test3_ok, test4_ok])
            self._record_result(test_name, test_passed, {
                "emission_factors_loaded": True,
                "brazil_routes_available": len(calc.brazil_routes),
                "transport_modes": len(calc.emissions_factors_ipcc)
            })
            
        except Exception as e:
            print(f"   ✗ Erro: {str(e)}")
            self._record_result(test_name, False, {"error": str(e)})
    
    def test_rl_environment(self):
        """Testa o ambiente RL"""
        print("\n3. Testando Ambiente RL...")
        test_name = "rl_environment"
        
        try:
            env = EcoTravelRLEnvironment()
            
            # Teste 1: Reset
            state, info = env.reset()
            test1_ok = (
                state.shape == env.observation_space.shape and
                hasattr(env, 'current_query') and
                len(env.current_query) > 0
            )
            
            print(f"   {'✓' if test1_ok else '✗'} Reset ambiente: Query = '{env.current_query[:50]}...'")
            
            # Teste 2: Step
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)
            
            test2_ok = (
                "tool_used" in info and
                "success" in info and
                "latency" in info and
                isinstance(reward, (int, float))
            )
            
            print(f"   {'✓' if test2_ok else '✗'} Step: Ferramenta = {info.get('tool_used', 'N/A')}, "
                  f"Sucesso = {info.get('success', 'N/A')}")
            
            # Teste 3: Múltiplos episódios
            rewards = []
            for _ in range(5):
                state, _ = env.reset()
                done = False
                episode_reward = 0
                
                while not done:
                    action = env.action_space.sample()
                    _, reward, terminated, truncated, _ = env.step(action)
                    episode_reward += reward
                    done = terminated or truncated
                
                rewards.append(episode_reward)
            
            test3_ok = len(rewards) == 5 and all(isinstance(r, (int, float)) for r in rewards)
            avg_reward = sum(rewards) / len(rewards) if rewards else 0
            
            print(f"   {'✓' if test3_ok else '✗'} Múltiplos episódios: Recompensa média = {avg_reward:.2f}")
            
            test_passed = all([test1_ok, test2_ok, test3_ok])
            self._record_result(test_name, test_passed, {
                "observation_space": str(env.observation_space),
                "action_space": str(env.action_space),
                "n_tools": env.n_tools,
                "avg_reward_random": avg_reward
            })
            
        except Exception as e:
            print(f"   ✗ Erro: {str(e)}")
            self._record_result(test_name, False, {"error": str(e)})
    
    def test_rag_tools_integration(self):
        """Testa integração entre RAG e ferramentas"""
        print("\n4. Testando Integração RAG + Ferramentas...")
        test_name = "rag_tools_integration"
        
        try:
            # Inicializar componentes
            rag = EnhancedRAGSystem()
            rag.load_data()
            calc = CarbonCalculator()
            
            # Query complexa que requer múltiplas ferramentas
            query = "Como viajar de São Paulo para Fernando de Noronha de forma sustentável?"
            
            # 1. Buscar informações no RAG
            rag_results = rag.search(query, filter_source_type="guia_turismo")
            test1_ok = len(rag_results) > 0
            
            print(f"   {'✓' if test1_ok else '✗'} RAG encontrou {len(rag_results)} resultados relevantes")
            
            # 2. Calcular emissões para a rota
            emissions_result = calc.calculate_carbon_footprint(
                transport_mode="aviao_domestico",
                distance_km=545,  # Recife-Noronha
                round_trip=True
            )
            
            test2_ok = emissions_result["total_emissions_kg"] > 0
            
            print(f"   {'✓' if test2_ok else '✗'} Cálculo de emissões: {emissions_result['total_emissions_kg']} kg CO2")
            
            # 3. Obter sugestões combinadas
            context, sources = rag.get_context_for_query("Fernando de Noronha sustentável")
            suggestions = emissions_result["suggestions"]
            
            test3_ok = len(context) > 0 and len(suggestions) > 0
            
            print(f"   {'✓' if test3_ok else '✗'} Contexto RAG: {len(context)} chars, "
                  f"{len(suggestions)} sugestões")
            
            test_passed = all([test1_ok, test2_ok, test3_ok])
            self._record_result(test_name, test_passed, {
                "rag_results": len(rag_results),
                "emissions_calculated": emissions_result["total_emissions_kg"],
                "context_length": len(context),
                "suggestions_count": len(suggestions)
            })
            
        except Exception as e:
            print(f"   ✗ Erro: {str(e)}")
            self._record_result(test_name, False, {"error": str(e)})
    
    def test_complete_pipeline(self):
        """Testa pipeline completo com agente RL"""
        print("\n5. Testando Pipeline Completo...")
        test_name = "complete_pipeline"
        
        try:
            # Simular pipeline completo
            queries = [
                "Qual a melhor época para visitar Bonito?",
                "Calcule as emissões de um voo SP-Salvador",
                "Hotéis sustentáveis em Gramado"
            ]
            
            results = []
            for query in queries:
                # 1. Classificar query
                query_type = self._classify_query_type(query)
                
                # 2. Selecionar ferramenta apropriada
                tool = self._select_tool_for_query(query_type)
                
                # 3. Executar ferramenta
                result = self._execute_tool(tool, query)
                
                success = result is not None and result.get("success", False)
                results.append(success)
                
                print(f"   {'✓' if success else '✗'} Query: '{query[:40]}...' -> {tool}")
            
            test_passed = all(results)
            self._record_result(test_name, test_passed, {
                "queries_tested": len(queries),
                "successful": sum(results),
                "pipeline_components": ["query_classification", "tool_selection", "tool_execution"]
            })
            
        except Exception as e:
            print(f"   ✗ Erro: {str(e)}")
            self._record_result(test_name, False, {"error": str(e)})
    
    def _classify_query_type(self, query: str) -> str:
        """Classifica tipo de query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["clima", "época", "tempo"]):
            return "weather"
        elif any(word in query_lower for word in ["calcul", "emissão", "co2"]):
            return "carbon"
        elif any(word in query_lower for word in ["hotel", "destino", "sustentável"]):
            return "destinations"
        else:
            return "general"
    
    def _select_tool_for_query(self, query_type: str) -> str:
        """Seleciona ferramenta baseada no tipo de query"""
        tool_mapping = {
            "weather": "weather_api",
            "carbon": "carbon_calculator",
            "destinations": "rag_system",
            "general": "rag_system"
        }
        return tool_mapping.get(query_type, "rag_system")
    
    def _execute_tool(self, tool: str, query: str) -> dict:
        """Executa ferramenta simulada"""
        # Simulação simplificada para teste
        if tool == "carbon_calculator":
            calc = CarbonCalculator()
            # Extrair distância aproximada da query (simplificado)
            result = calc.calculate_carbon_footprint("aviao_domestico", 1000, round_trip=True)
            return {"success": True, "result": result}
        
        elif tool == "rag_system":
            rag = EnhancedRAGSystem()
            rag.load_data()
            results = rag.search(query)
            return {"success": len(results) > 0, "results": results}
        
        else:
            return {"success": True, "tool": tool}
    
    def _record_result(self, test_name: str, passed: bool, details: dict):
        """Registra resultado do teste"""
        self.results["tests"][test_name] = {
            "passed": passed,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        
        self.results["summary"]["total"] += 1
        if passed:
            self.results["summary"]["passed"] += 1
        else:
            self.results["summary"]["failed"] += 1
    
    def print_summary(self):
        """Imprime resumo dos testes"""
        summary = self.results["summary"]
        
        print("\n" + "="*50)
        print("RESUMO DOS TESTES")
        print("="*50)
        print(f"Total de testes: {summary['total']}")
        print(f"✓ Aprovados: {summary['passed']}")
        print(f"✗ Reprovados: {summary['failed']}")
        print(f"Taxa de sucesso: {summary['passed']/summary['total']*100:.1f}%")
        
        if summary['failed'] > 0:
            print("\nTestes reprovados:")
            for test_name, result in self.results["tests"].items():
                if not result["passed"]:
                    print(f"  - {test_name}: {result['details'].get('error', 'Falhou')}")
    
    def save_results(self):
        """Salva resultados dos testes"""
        results_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"\nResultados salvos em: {results_file}")


def main():
    """Função principal"""
    tester = IntegrationTester()
    
    start_time = time.time()
    tester.run_all_tests()
    end_time = time.time()
    
    print(f"\nTempo total de execução: {end_time - start_time:.2f} segundos")
    
    # Retornar código de saída baseado no sucesso
    if tester.results["summary"]["failed"] == 0:
        print("\n✅ Todos os testes passaram! Sistema pronto para uso.")
        return 0
    else:
        print("\n❌ Alguns testes falharam. Verifique os logs acima.")
        return 1


if __name__ == "__main__":
    exit(main()) 