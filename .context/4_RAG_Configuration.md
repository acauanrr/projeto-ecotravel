# Configuração do RAG

## Base de Conhecimento
- **Fontes**:
  - Guias de viagem sustentáveis (PDFs).
  - Tabelas de emissões de CO2 (CSVs).
  - Regulamentações ambientais (TXTs).
  - Avaliações de hotéis eco-friendly (JSONs).
- **Formato**: Arquivos armazenados em `data/` com estrutura organizada.

## Estratégias Modernas
- **Chunking Inteligente**:
  - Método: Semantic (RecursiveCharacterTextSplitter).
  - Configuração: Chunk size = 512, overlap = 50.
- **Embeddings**:
  - Modelo: `sentence-transformers/all-MiniLM-L6-v2` (leve, 384 dimensões).
  - Armazenamento: FAISS (vetor store CPU-friendly).
- **Recuperação**:
  - Hybrid Search: Combina BM25 (Rank-BM25) com busca semântica.
  - Top-k: 10 documentos.
  - Reranking: Cohere Rerank (API gratuita limitada) ou BGE-reranker local.
- **Anti-alucinação**:
  - Self-Query: Reformula queries para maior precisão.
  - Fact-Checking: Verifica respostas contra fontes.
  - Source Citation: Inclui referências na saída.

## Configuração
```python
rag_config = {
    "chunking": {
        "method": "semantic",
        "size": 512,
        "overlap": 50
    },
    "embeddings": {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "dimension": 384
    },
    "retrieval": {
        "hybrid_search": True,
        "top_k": 10,
        "reranking": True
    },
    "anti_hallucination": {
        "self_query": True,
        "fact_checking": True,
        "source_citation": True
    }
}
```

## Implementação Inicial
```python
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Configurar chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " "]
)

# Carregar documentos
documents = SimpleDirectoryReader("data/").load_data()

# Criar índice
index = VectorStoreIndex.from_documents(documents, text_splitter=splitter)
```

## Validação
- **Métricas**: Hit Rate, MRR (Mean Reciprocal Rank).
- **Testes**: Queries simuladas com ground truth (ex.: "melhores hotéis eco-friendly no RJ").