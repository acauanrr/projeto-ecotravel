#!/usr/bin/env python3
"""
Script de teste para verificar instalação do EcoTravel Agent
"""

import sys
import importlib
from typing import List, Tuple

def test_import(module_name: str, package_name: str = None) -> Tuple[bool, str]:
    """Testa import de um módulo"""
    if package_name is None:
        package_name = module_name
    
    try:
        importlib.import_module(module_name)
        return True, f"✅ {package_name}"
    except ImportError as e:
        return False, f"❌ {package_name}: {str(e)}"

def main():
    print("🧪 Testando instalação do EcoTravel Agent...")
    print("="*50)
    
    # Lista de módulos essenciais
    essential_modules = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("openai", "OpenAI"),
        ("langchain", "LangChain"),
        ("gymnasium", "Gymnasium"),
        ("stable_baselines3", "Stable-Baselines3"),
        ("transformers", "Transformers"),
        ("sentence_transformers", "Sentence-Transformers"),
        ("faiss", "FAISS"),
        ("streamlit", "Streamlit"),
        ("plotly", "Plotly"),
        ("requests", "Requests")
    ]
    
    # Lista de módulos opcionais
    optional_modules = [
        ("duckduckgo_search", "DuckDuckGo Search"),
        ("googleapiclient", "Google API Client"),
        ("pypdf", "PyPDF"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn")
    ]
    
    print("📦 Módulos Essenciais:")
    essential_failures = []
    for module, name in essential_modules:
        success, message = test_import(module, name)
        print(f"   {message}")
        if not success:
            essential_failures.append(name)
    
    print("\n🔧 Módulos Opcionais:")
    optional_failures = []
    for module, name in optional_modules:
        success, message = test_import(module, name)
        print(f"   {message}")
        if not success:
            optional_failures.append(name)
    
    # Testar funcionalidades específicas
    print("\n🔍 Testando funcionalidades específicas:")
    
    # Testar PyTorch
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   ✅ PyTorch disponível em: {device}")
    except:
        print("   ❌ PyTorch não funcionando corretamente")
    
    # Testar OpenAI
    try:
        import openai
        print("   ✅ OpenAI configurado")
    except:
        print("   ⚠️ OpenAI não configurado (configure OPENAI_API_KEY)")
    
    # Testar ambiente RL
    try:
        import gymnasium as gym
        env = gym.make("CartPole-v1")
        env.close()
        print("   ✅ Gymnasium funcionando")
    except Exception as e:
        print(f"   ❌ Gymnasium com problema: {e}")
    
    # Resumo
    print("\n" + "="*50)
    print("📊 RESUMO DOS TESTES:")
    
    if not essential_failures:
        print("🎉 Todos os módulos essenciais estão funcionando!")
        print("✅ O EcoTravel Agent está pronto para uso!")
    else:
        print(f"⚠️ {len(essential_failures)} módulos essenciais falharam:")
        for failure in essential_failures:
            print(f"   - {failure}")
        print("\n💡 Execute: python install_dependencies.py")
    
    if optional_failures:
        print(f"\nℹ️ {len(optional_failures)} módulos opcionais não estão disponíveis:")
        for failure in optional_failures:
            print(f"   - {failure}")
        print("   (Isso não impede o funcionamento básico)")
    
    # Próximos passos
    print("\n📋 Próximos Passos:")
    if not essential_failures:
        print("1. Configure suas API keys:")
        print("   export OPENAI_API_KEY='sua-chave'")
        print("2. Execute o demo:")
        print("   python demo_ecotravel.py")
        print("3. Ou abra o notebook:")
        print("   jupyter notebook notebooks/EcoTravel_Agent_RL_Colab.ipynb")
    else:
        print("1. Resolva os problemas de instalação")
        print("2. Execute: python install_dependencies.py")
        print("3. Ou use o Google Colab para evitar problemas locais")
    
    return len(essential_failures) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 