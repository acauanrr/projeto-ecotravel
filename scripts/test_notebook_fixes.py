#!/usr/bin/env python3
"""
Script de teste para verificar se as correções do notebook estão funcionando
"""

import sys
import os

# Adicionar diretório raiz ao path
sys.path.insert(0, os.path.abspath('.'))

def test_imports():
    """Testa se todos os imports estão funcionando"""
    print("🧪 Testando imports...")
    
    try:
        from src.rag.enhanced_rag_system import EnhancedRAGSystem
        print("✓ EnhancedRAGSystem importado com sucesso")
    except Exception as e:
        print(f"✗ Erro ao importar EnhancedRAGSystem: {e}")
        return False
    
    try:
        from src.tools.carbon_calculator import CarbonCalculator
        print("✓ CarbonCalculator importado com sucesso")
    except Exception as e:
        print(f"✗ Erro ao importar CarbonCalculator: {e}")
        return False
    
    try:
        from src.rl.environment import EcoTravelRLEnvironment
        print("✓ EcoTravelRLEnvironment importado com sucesso")
    except Exception as e:
        print(f"✗ Erro ao importar EcoTravelRLEnvironment: {e}")
        return False
    
    return True

def test_rag_system():
    """Testa o sistema RAG"""
    print("\n🔍 Testando sistema RAG...")
    
    try:
        from src.rag.enhanced_rag_system import EnhancedRAGSystem
        rag = EnhancedRAGSystem()
        
        # Verificar se tem método get_context_for_query
        if hasattr(rag, 'get_context_for_query'):
            print("✓ Método get_context_for_query disponível")
        else:
            print("✗ Método get_context_for_query não encontrado")
            return False
        
        # Tentar carregar dados
        rag.load_data()
        print("✓ Dados carregados com sucesso")
        
        # Testar busca
        results = rag.search("turismo sustentável")
        print(f"✓ Busca funcionando: {len(results)} resultados")
        
        return True
        
    except Exception as e:
        print(f"✗ Erro no teste RAG: {e}")
        return False

def test_carbon_calculator():
    """Testa a calculadora de carbono"""
    print("\n💨 Testando calculadora de carbono...")
    
    try:
        from src.tools.carbon_calculator import CarbonCalculator
        calc = CarbonCalculator()
        
        # Testar cálculo
        result = calc.calculate_carbon_footprint(
            transport_mode="aviao_domestico",
            distance_km=500,
            round_trip=True
        )
        
        print(f"✓ Cálculo funcionando: {result['total_emissions_kg']} kg CO2")
        return True
        
    except Exception as e:
        print(f"✗ Erro no teste da calculadora: {e}")
        return False

def test_rl_environment():
    """Testa o ambiente RL"""
    print("\n🤖 Testando ambiente RL...")
    
    try:
        from src.rl.environment import EcoTravelRLEnvironment
        env = EcoTravelRLEnvironment(use_advanced_embeddings=False)
        
        # Testar reset
        state, info = env.reset()
        print(f"✓ Reset funcionando: estado shape {state.shape}")
        
        # Testar step
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        print(f"✓ Step funcionando: ferramenta {info['tool_used']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Erro no teste RL: {e}")
        return False

def main():
    """Executa todos os testes"""
    print("🔧 Executando testes das correções do notebook\n")
    
    tests = [
        test_imports,
        test_rag_system,
        test_carbon_calculator,
        test_rl_environment
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print("❌ Teste falhou")
        except Exception as e:
            print(f"❌ Erro no teste: {e}")
    
    print(f"\n📊 Resultado: {passed}/{total} testes passaram")
    
    if passed == total:
        print("✅ Todas as correções estão funcionando!")
        print("🎯 O notebook deve executar sem erros agora.")
    else:
        print("⚠️ Algumas correções ainda precisam de atenção.")
        print("📝 Verifique os erros acima e execute novamente.")

if __name__ == "__main__":
    main()