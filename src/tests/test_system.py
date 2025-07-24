"""
Testes e métricas de validação para o EcoTravel Agent
"""

import sys
import os
import time
import json
import unittest
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import numpy as np

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent.parent))

# Importar componentes para teste
try:
    from tools.carbon_calculator import CarbonCalculator
    from tools.weather_api import WeatherAPI
    from tools.web_search import WebSearchTool
    TOOLS_AVAILABLE = True
except ImportError as e:
    print(f"Ferramentas não disponíveis: {e}")
    TOOLS_AVAILABLE = False

try:
    from rag.rag_system import AdvancedRAGSystem
    RAG_AVAILABLE = True
except ImportError as e:
    print(f"Sistema RAG não disponível: {e}")
    RAG_AVAILABLE = False

try:
    from agent.eco_travel_agent import EcoTravelAgent
    AGENT_AVAILABLE = True
except ImportError as e:
    print(f"Agente não disponível: {e}")
    AGENT_AVAILABLE = False


class TestCarbonCalculator(unittest.TestCase):
    """Testes para a calculadora de carbono"""
    
    def setUp(self):
        if not TOOLS_AVAILABLE:
            self.skipTest("Ferramentas não disponíveis")
        self.calc = CarbonCalculator()
    
    def test_emission_factor_retrieval(self):
        """Testa obtenção de fatores de emissão"""
        # Testa fatores conhecidos
        self.assertAlmostEqual(self.calc.get_emission_factor("aviao"), 0.255, places=3)
        self.assertAlmostEqual(self.calc.get_emission_factor("onibus"), 0.089, places=3)
        self.assertAlmostEqual(self.calc.get_emission_factor("trem"), 0.041, places=3)
        
        # Testa modal inexistente
        self.assertEqual(self.calc.get_emission_factor("foguete"), 0.0)
    
    def test_carbon_footprint_calculation(self):
        """Testa cálculo de pegada de carbono"""
        # Teste básico
        result = self.calc.calculate_carbon_footprint("aviao", 400)
        expected_emission = 0.255 * 400  # fator * distância
        self.assertAlmostEqual(result["total_emission_kg_co2"], expected_emission, places=2)
        
        # Teste ida e volta
        result_round = self.calc.calculate_carbon_footprint("aviao", 400, round_trip=True)
        self.assertAlmostEqual(result_round["total_emission_kg_co2"], expected_emission * 2, places=2)
    
    def test_transport_mode_comparison(self):
        """Testa comparação entre modais"""
        comparison = self.calc.compare_transport_modes(400)
        
        # Deve retornar lista ordenada por emissão
        self.assertIsInstance(comparison, list)
        self.assertGreater(len(comparison), 0)
        
        # Verificar ordenação (menor para maior emissão)
        for i in range(len(comparison) - 1):
            self.assertLessEqual(
                comparison[i]["total_emission_kg_co2"],
                comparison[i + 1]["total_emission_kg_co2"]
            )
    
    def test_sustainability_recommendation(self):
        """Testa geração de recomendações"""
        rec = self.calc.get_sustainability_recommendation("SP", "RJ", 400)
        
        # Verificar estrutura da recomendação
        self.assertIn("most_sustainable_mode", rec)
        self.assertIn("emission_savings_kg_co2", rec)
        self.assertIn("recommendation", rec)
        
        # Verificar que economia é não-negativa
        self.assertGreaterEqual(rec["emission_savings_kg_co2"], 0)


class TestWeatherAPI(unittest.TestCase):
    """Testes para a API de clima"""
    
    def setUp(self):
        if not TOOLS_AVAILABLE:
            self.skipTest("Ferramentas não disponíveis")
        self.weather = WeatherAPI()
    
    def test_coordinates_retrieval(self):
        """Testa obtenção de coordenadas"""
        coords = self.weather.get_coordinates("São Paulo")
        
        if coords:  # Se a API está funcionando
            self.assertIsInstance(coords, tuple)
            self.assertEqual(len(coords), 2)
            
            # Verificar se as coordenadas estão no range válido
            lat, lon = coords
            self.assertGreaterEqual(lat, -90)
            self.assertLessEqual(lat, 90)
            self.assertGreaterEqual(lon, -180)
            self.assertLessEqual(lon, 180)
    
    def test_weather_data_structure(self):
        """Testa estrutura dos dados meteorológicos"""
        coords = self.weather.get_coordinates("São Paulo")
        
        if coords:
            current = self.weather.get_current_weather(coords[0], coords[1])
            
            if current:  # Se a API retornou dados
                expected_keys = [
                    "temperature_celsius", "humidity_percent", 
                    "weather_description", "wind_speed_kmh"
                ]
                
                for key in expected_keys:
                    self.assertIn(key, current)


class TestWebSearch(unittest.TestCase):
    """Testes para busca web"""
    
    def setUp(self):
        if not TOOLS_AVAILABLE:
            self.skipTest("Ferramentas não disponíveis")
        self.search = WebSearchTool()
    
    def test_search_structure(self):
        """Testa estrutura dos resultados de busca"""
        results = self.search.search_web("teste", num_results=1)
        
        self.assertIsInstance(results, list)
        
        if results and not results[0].get("error"):
            result = results[0]
            expected_keys = ["title", "url", "snippet", "source"]
            
            for key in expected_keys:
                self.assertIn(key, result)


class TestRAGSystem(unittest.TestCase):
    """Testes para o sistema RAG"""
    
    def setUp(self):
        if not RAG_AVAILABLE:
            self.skipTest("Sistema RAG não disponível")
        
        # Usar dados de teste
        test_data_path = Path(__file__).parent.parent.parent / "data"
        self.rag = AdvancedRAGSystem(data_path=str(test_data_path))
    
    def test_document_loading(self):
        """Testa carregamento de documentos"""
        documents = self.rag.load_documents()
        
        self.assertIsInstance(documents, list)
        self.assertGreater(len(documents), 0)
        
        # Verificar estrutura dos documentos
        for doc in documents:
            self.assertTrue(hasattr(doc, 'page_content'))
            self.assertTrue(hasattr(doc, 'metadata'))
    
    def test_chunking_process(self):
        """Testa processo de chunking"""
        self.rag.load_documents()
        chunks = self.rag.create_chunks()
        
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
        
        # Verificar que chunks não são muito grandes
        for chunk in chunks:
            self.assertLessEqual(len(chunk.page_content), self.rag.chunk_size * 1.2)
    
    def test_search_functionality(self):
        """Testa funcionalidade de busca"""
        # Construir índice se necessário
        if not self.rag.chunks:
            self.rag.build_index()
        
        if self.rag.chunks:
            # Teste de busca semântica
            results = self.rag.search("transporte sustentável", method="semantic", k=3)
            
            self.assertIsInstance(results, list)
            self.assertLessEqual(len(results), 3)
            
            # Verificar estrutura dos resultados
            for result in results:
                self.assertIn("content", result)
                self.assertIn("score", result)
                self.assertIn("metadata", result)


class TestEcoTravelAgent(unittest.TestCase):
    """Testes para o agente principal"""
    
    def setUp(self):
        if not AGENT_AVAILABLE:
            self.skipTest("Agente não disponível")
        
        try:
            self.agent = EcoTravelAgent(
                data_path=str(Path(__file__).parent.parent.parent / "data"),
                verbose=False
            )
        except Exception as e:
            self.skipTest(f"Erro ao inicializar agente: {e}")
    
    def test_agent_initialization(self):
        """Testa inicialização do agente"""
        self.assertIsNotNone(self.agent)
        self.assertIsNotNone(self.agent.tools)
        self.assertGreater(len(self.agent.tools), 0)
    
    def test_sustainability_scoring(self):
        """Testa sistema de scoring de sustentabilidade"""
        test_plan = {
            "transport_mode": "onibus",
            "distance_km": 400,
            "duration_days": 3,
            "eco_hotel": True,
            "local_activities": True,
            "passengers": 2
        }
        
        score_result = self.agent.get_sustainability_score(test_plan)
        
        self.assertIn("score", score_result)
        self.assertIn("category", score_result)
        self.assertIn("factors", score_result)
        self.assertIn("recommendations", score_result)
        
        # Score deve estar entre 0 e 100
        self.assertGreaterEqual(score_result["score"], 0)
        self.assertLessEqual(score_result["score"], 100)


class SystemBenchmark:
    """Benchmark completo do sistema"""
    
    def __init__(self):
        self.results = {}
    
    def run_carbon_calculator_benchmark(self):
        """Benchmark da calculadora de carbono"""
        if not TOOLS_AVAILABLE:
            return {"error": "Ferramentas não disponíveis"}
        
        calc = CarbonCalculator()
        
        # Teste de performance
        distances = [100, 400, 800, 1200, 1600]
        modes = ["aviao", "carro", "onibus", "trem"]
        
        start_time = time.time()
        calculations = 0
        
        for distance in distances:
            for mode in modes:
                calc.calculate_carbon_footprint(mode, distance)
                calculations += 1
        
        end_time = time.time()
        
        return {
            "total_calculations": calculations,
            "total_time": end_time - start_time,
            "calculations_per_second": calculations / (end_time - start_time),
            "avg_time_per_calculation": (end_time - start_time) / calculations
        }
    
    def run_rag_benchmark(self):
        """Benchmark do sistema RAG"""
        if not RAG_AVAILABLE:
            return {"error": "Sistema RAG não disponível"}
        
        try:
            test_data_path = Path(__file__).parent.parent.parent / "data"
            rag = AdvancedRAGSystem(data_path=str(test_data_path))
            
            # Benchmark de construção
            start_time = time.time()
            rag.build_index()
            build_time = time.time() - start_time
            
            # Benchmark de busca
            test_queries = [
                "transporte sustentável",
                "hotéis ecológicos",
                "emissão carbono",
                "energia solar",
                "atividades locais"
            ]
            
            search_times = []
            for query in test_queries:
                start_time = time.time()
                results = rag.search(query, k=5)
                search_time = time.time() - start_time
                search_times.append(search_time)
            
            return {
                "build_time": build_time,
                "total_documents": len(rag.documents),
                "total_chunks": len(rag.chunks),
                "avg_search_time": np.mean(search_times),
                "searches_per_second": 1 / np.mean(search_times),
                "embedding_dimension": rag.embeddings.shape[1] if rag.embeddings is not None else 0
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def run_full_benchmark(self):
        """Executa benchmark completo"""
        print("🚀 Executando Benchmark Completo do Sistema")
        print("=" * 60)
        
        # Benchmark da calculadora
        print("\n📊 Benchmark da Calculadora de Carbono...")
        carbon_results = self.run_carbon_calculator_benchmark()
        self.results["carbon_calculator"] = carbon_results
        
        if "error" not in carbon_results:
            print(f"   ✅ {carbon_results['total_calculations']} cálculos em {carbon_results['total_time']:.3f}s")
            print(f"   ⚡ {carbon_results['calculations_per_second']:.1f} cálculos/segundo")
        else:
            print(f"   ❌ {carbon_results['error']}")
        
        # Benchmark do RAG
        print("\n🧠 Benchmark do Sistema RAG...")
        rag_results = self.run_rag_benchmark()
        self.results["rag_system"] = rag_results
        
        if "error" not in rag_results:
            print(f"   ✅ Índice construído em {rag_results['build_time']:.2f}s")
            print(f"   📚 {rag_results['total_documents']} documentos, {rag_results['total_chunks']} chunks")
            print(f"   🔍 Busca média: {rag_results['avg_search_time']:.3f}s ({rag_results['searches_per_second']:.1f} buscas/s)")
        else:
            print(f"   ❌ {rag_results['error']}")
        
        # Salvar resultados
        results_path = Path(__file__).parent.parent.parent / "benchmark_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Resultados salvos em: {results_path}")
        
        return self.results


def run_all_tests():
    """Executa todos os testes"""
    print("🧪 Executando Testes do Sistema EcoTravel")
    print("=" * 50)
    
    # Configurar suite de testes
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Adicionar classes de teste
    test_classes = [
        TestCarbonCalculator,
        TestWeatherAPI, 
        TestWebSearch,
        TestRAGSystem,
        TestEcoTravelAgent
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Executar testes
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Resumo
    print("\n" + "=" * 50)
    print("📋 Resumo dos Testes:")
    print(f"   ✅ Sucessos: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"   ❌ Falhas: {len(result.failures)}")
    print(f"   🚫 Erros: {len(result.errors)}")
    print(f"   ⏭️ Pulados: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    return result.wasSuccessful()


def generate_test_report():
    """Gera relatório completo de testes e benchmark"""
    print("📄 Gerando Relatório de Testes e Performance")
    print("=" * 60)
    
    # Executar testes
    test_success = run_all_tests()
    
    # Executar benchmark
    benchmark = SystemBenchmark()
    benchmark_results = benchmark.run_full_benchmark()
    
    # Gerar relatório
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tests_passed": test_success,
        "benchmark_results": benchmark_results,
        "system_info": {
            "tools_available": TOOLS_AVAILABLE,
            "rag_available": RAG_AVAILABLE,
            "agent_available": AGENT_AVAILABLE
        }
    }
    
    # Salvar relatório
    report_path = Path(__file__).parent.parent.parent / "test_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📊 Relatório completo salvo em: {report_path}")
    
    return report


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Testes do EcoTravel Agent")
    parser.add_argument("--benchmark", action="store_true", help="Executar apenas benchmark")
    parser.add_argument("--tests", action="store_true", help="Executar apenas testes")
    parser.add_argument("--report", action="store_true", help="Gerar relatório completo")
    
    args = parser.parse_args()
    
    if args.benchmark:
        benchmark = SystemBenchmark()
        benchmark.run_full_benchmark()
    elif args.tests:
        run_all_tests()
    elif args.report:
        generate_test_report()
    else:
        # Executar relatório completo por padrão
        generate_test_report()