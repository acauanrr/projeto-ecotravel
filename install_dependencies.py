#!/usr/bin/env python3
"""
Script de instalação de dependências do EcoTravel Agent
Resolve conflitos de versão automaticamente
"""

import subprocess
import sys
import os

def run_command(cmd):
    """Executa comando e retorna saída"""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Erro: {e.stderr}")
        return None

def main():
    print("🔧 EcoTravel Agent - Instalação de Dependências")
    print("=" * 50)
    
    # 1. Limpar instalações conflitantes
    print("\n📦 Limpando versões conflitantes do LangChain...")
    packages_to_uninstall = [
        "langchain",
        "langchain-core",
        "langchain-community",
        "langchain-text-splitters",
        "langchain-openai",
        "langchain-experimental",
        "langsmith"
    ]
    
    for package in packages_to_uninstall:
        run_command([sys.executable, "-m", "pip", "uninstall", "-y", package])
    
    print("✅ Limpeza concluída!")
    
    # 2. Instalar LangChain com versões compatíveis
    print("\n📦 Instalando LangChain com versões compatíveis...")
    
    # Opção 1: Versões estáveis e testadas
    langchain_deps = [
        "langchain==0.2.6",
        "langchain-community==0.2.6",
        "langchain-core==0.2.10",
        "langchain-text-splitters==0.2.2",
        "langsmith==0.1.83"
    ]
    
    for package in langchain_deps:
        print(f"  → Instalando {package}...")
        result = run_command([sys.executable, "-m", "pip", "install", package])
        if result is None:
            print(f"  ✗ Erro ao instalar {package}")
            return 1
    
    print("✅ LangChain instalado com sucesso!")
    
    # 3. Instalar outras dependências essenciais
    print("\n📦 Instalando outras dependências...")
    
    other_deps = [
        "openai>=1.0.0",
        "sentence-transformers>=2.2.0",
        "transformers>=4.30.0",
        "faiss-cpu>=1.7.4",
        "rank-bm25>=0.2.2",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "gymnasium>=0.29.0",
        "stable-baselines3>=2.0.0",
        "torch>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.14.0",
        "requests>=2.31.0",
        "beautifulsoup4>=4.12.0",
        "duckduckgo-search>=3.8.0",
        "streamlit>=1.25.0",
        "python-dotenv>=1.0.0",
        "tqdm>=4.65.0"
    ]
    
    for package in other_deps:
        print(f"  → Instalando {package}...")
        result = run_command([sys.executable, "-m", "pip", "install", "-q", package])
        if result is None:
            print(f"  ✗ Erro ao instalar {package}")
    
    print("\n✅ Instalação concluída!")
    
    # 4. Verificar instalações
    print("\n🔍 Verificando instalações...")
    
    test_imports = [
        ("langchain", "LangChain"),
        ("langchain_community", "LangChain Community"),
        ("sentence_transformers", "Sentence Transformers"),
        ("faiss", "FAISS"),
        ("gymnasium", "Gymnasium"),
        ("stable_baselines3", "Stable Baselines3")
    ]
    
    all_success = True
    for module, name in test_imports:
        try:
            __import__(module)
            print(f"  ✓ {name} importado com sucesso")
        except ImportError as e:
            print(f"  ✗ Erro ao importar {name}: {e}")
            all_success = False
    
    if all_success:
        print("\n🎉 Todas as dependências foram instaladas com sucesso!")
        print("\n📝 Próximos passos:")
        print("   1. Execute o notebook EcoTravel_Agent_RL_Local.ipynb")
        print("   2. Ou execute: jupyter notebook EcoTravel_Agent_RL_Local.ipynb")
        return 0
    else:
        print("\n⚠️ Algumas dependências não foram instaladas corretamente.")
        print("   Tente executar o script novamente ou instale manualmente.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 