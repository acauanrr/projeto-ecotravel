#!/usr/bin/env python3
"""
Script de instalação de dependências para EcoTravel Agent
Resolve problemas de compatibilidade de versões
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Executa comando e trata erros"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} - Sucesso!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Erro:")
        print(f"   {e.stderr}")
        return False

def install_package(package, description=None):
    """Instala pacote individual"""
    if description is None:
        description = f"Instalando {package}"
    return run_command(f"pip install {package}", description)

def main():
    print("🚀 Instalando dependências do EcoTravel Agent...")
    print("="*60)
    
    # Verificar versão do Python
    python_version = sys.version_info
    print(f"🐍 Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("❌ Python 3.8+ é necessário!")
        return False
    
    # Instalar dependências básicas primeiro
    basic_packages = [
        "numpy>=1.21.0",
        "pandas>=2.0.0", 
        "requests>=2.25.0",
        "python-dotenv>=0.19.0"
    ]
    
    for package in basic_packages:
        if not install_package(package):
            print(f"⚠️ Falha ao instalar {package}, continuando...")
    
    # Instalar PyTorch (versão compatível)
    print("\n🔥 Instalando PyTorch...")
    if python_version >= (3, 12):
        # Python 3.12+ - usar versão mais recente
        torch_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
    else:
        # Python 3.8-3.11 - usar versão estável
        torch_cmd = "pip install torch>=2.2.0 torchvision>=0.17.0 torchaudio>=2.2.0"
    
    if not run_command(torch_cmd, "Instalando PyTorch"):
        print("⚠️ Tentando instalar PyTorch sem versão específica...")
        install_package("torch", "Instalando PyTorch (versão padrão)")
    
    # Instalar bibliotecas de ML/AI
    ml_packages = [
        "openai>=1.3.0",
        "transformers>=4.35.0",
        "sentence-transformers>=2.2.0",
        "faiss-cpu>=1.7.0",
        "gymnasium>=0.29.0",
        "stable-baselines3>=2.2.0"
    ]
    
    print("\n🤖 Instalando bibliotecas de ML/AI...")
    for package in ml_packages:
        if not install_package(package):
            print(f"⚠️ Falha ao instalar {package}, continuando...")
    
    # Instalar LangChain e relacionados
    langchain_packages = [
        "langchain>=0.0.350",
        "langchain-community>=0.0.1",
        "langchain-experimental>=0.0.43",
        "llama-index>=0.9.0"
    ]
    
    print("\n🔗 Instalando LangChain...")
    for package in langchain_packages:
        if not install_package(package):
            print(f"⚠️ Falha ao instalar {package}, continuando...")
    
    # Instalar ferramentas web
    web_packages = [
        "duckduckgo-search>=3.9.0",
        "google-api-python-client>=2.100.0",
        "rank-bm25>=0.2.0"
    ]
    
    print("\n🌐 Instalando ferramentas web...")
    for package in web_packages:
        if not install_package(package):
            print(f"⚠️ Falha ao instalar {package}, continuando...")
    
    # Instalar processamento de documentos
    doc_packages = [
        "pypdf>=3.15.0",
        "pdfplumber>=0.10.0",
        "python-docx>=1.1.0",
        "openpyxl>=3.1.0"
    ]
    
    print("\n📄 Instalando processamento de documentos...")
    for package in doc_packages:
        if not install_package(package):
            print(f"⚠️ Falha ao instalar {package}, continuando...")
    
    # Instalar visualização
    viz_packages = [
        "streamlit>=1.25.0",
        "plotly>=5.15.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0"
    ]
    
    print("\n📊 Instalando bibliotecas de visualização...")
    for package in viz_packages:
        if not install_package(package):
            print(f"⚠️ Falha ao instalar {package}, continuando...")
    
    # Instalar ferramentas de desenvolvimento
    dev_packages = [
        "jupyter>=1.0.0",
        "ipykernel>=6.20.0",
        "black>=23.0.0",
        "pylint>=3.0.0",
        "tqdm>=4.60.0",
        "colorama>=0.4.0"
    ]
    
    print("\n🛠️ Instalando ferramentas de desenvolvimento...")
    for package in dev_packages:
        if not install_package(package):
            print(f"⚠️ Falha ao instalar {package}, continuando...")
    
    # Verificar instalação
    print("\n🔍 Verificando instalação...")
    test_imports = [
        ("torch", "PyTorch"),
        ("langchain", "LangChain"),
        ("openai", "OpenAI"),
        ("gymnasium", "Gymnasium"),
        ("stable_baselines3", "Stable-Baselines3"),
        ("streamlit", "Streamlit"),
        ("plotly", "Plotly")
    ]
    
    failed_imports = []
    for module, name in test_imports:
        try:
            __import__(module)
            print(f"✅ {name} - OK")
        except ImportError:
            print(f"❌ {name} - Falhou")
            failed_imports.append(name)
    
    print("\n" + "="*60)
    if failed_imports:
        print(f"⚠️ Algumas dependências falharam: {', '.join(failed_imports)}")
        print("💡 Tente instalar manualmente ou use o Google Colab")
    else:
        print("🎉 Todas as dependências principais foram instaladas com sucesso!")
    
    print("\n📋 Próximos passos:")
    print("1. Configure suas API keys:")
    print("   export OPENAI_API_KEY='sua-chave'")
    print("   export GOOGLE_API_KEY='sua-chave'")
    print("2. Execute o demo: python demo_ecotravel.py")
    print("3. Ou abra o notebook: jupyter notebook notebooks/EcoTravel_Agent_RL_Colab.ipynb")
    
    return len(failed_imports) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 