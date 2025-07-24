from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ecotravel-agent",
    version="1.0.0",
    author="Seu Nome",
    author_email="seu.email@example.com",
    description="Assistente inteligente para planejamento de viagens sustentáveis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/seu-usuario/ecotravel-agent",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Other/Nonlisted Topic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.991",
            "pre-commit>=2.20.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "web": [
            "gradio>=4.0.0",
            "streamlit>=1.29.0",
        ],
        "all": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.991",
            "pre-commit>=2.20.0",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "gradio>=4.0.0",
            "streamlit>=1.29.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ecotravel=agent.eco_travel_agent:main",
            "ecotravel-test=tests.test_system:run_all_tests",
            "ecotravel-benchmark=tests.test_system:SystemBenchmark.run_full_benchmark",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["data/**/*"],
    },
    keywords="ai, travel, sustainability, carbon-footprint, rag, llm",
    project_urls={
        "Bug Reports": "https://github.com/seu-usuario/ecotravel-agent/issues",
        "Source": "https://github.com/seu-usuario/ecotravel-agent",
        "Documentation": "https://github.com/seu-usuario/ecotravel-agent/docs",
    },
)