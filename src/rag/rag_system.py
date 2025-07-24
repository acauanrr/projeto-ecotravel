"""
Sistema RAG avançado para o EcoTravel Agent
Implementa estratégias de chunking semântico, busca híbrida e reranking
"""

import os
import json
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import faiss

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, CSVLoader, TextLoader
from langchain.schema import Document
from rank_bm25 import BM25Okapi


class AdvancedRAGSystem:
    def __init__(
        self,
        data_path: str = "data",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        top_k: int = 10
    ):
        self.data_path = Path(data_path)
        self.embedding_model_name = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        
        # Inicializar componentes
        self.embedding_model = SentenceTransformer(embedding_model)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Armazenamento
        self.documents = []
        self.chunks = []
        self.embeddings = None
        self.faiss_index = None
        self.bm25 = None
        self.metadata = []
        
    def load_documents(self) -> List[Document]:
        """Carrega documentos de diferentes formatos"""
        documents = []
        
        # Carregar PDFs e TXTs dos guias
        guias_path = self.data_path / "guias"
        if guias_path.exists():
            loader = DirectoryLoader(
                str(guias_path),
                glob="**/*.txt",
                loader_cls=TextLoader
            )
            documents.extend(loader.load())
        
        # Carregar CSVs de emissões
        emissoes_path = self.data_path / "emissoes"
        if emissoes_path.exists():
            for csv_file in emissoes_path.glob("*.csv"):
                try:
                    df = pd.read_csv(csv_file)
                    content = df.to_string(index=False)
                    doc = Document(
                        page_content=content,
                        metadata={"source": str(csv_file), "type": "emissions_data"}
                    )
                    documents.append(doc)
                except Exception as e:
                    print(f"Erro ao carregar {csv_file}: {e}")
        
        # Carregar JSONs de avaliações
        avaliacoes_path = self.data_path / "avaliacoes"
        if avaliacoes_path.exists():
            for json_file in avaliacoes_path.glob("*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    content = json.dumps(data, indent=2, ensure_ascii=False)
                    doc = Document(
                        page_content=content,
                        metadata={"source": str(json_file), "type": "reviews"}
                    )
                    documents.append(doc)
                except Exception as e:
                    print(f"Erro ao carregar {json_file}: {e}")
        
        self.documents = documents
        return documents
    
    def create_chunks(self) -> List[Document]:
        """Cria chunks semânticos dos documentos"""
        chunks = []
        
        for doc in self.documents:
            doc_chunks = self.text_splitter.split_documents([doc])
            chunks.extend(doc_chunks)
        
        self.chunks = chunks
        return chunks
    
    def create_embeddings(self) -> np.ndarray:
        """Cria embeddings dos chunks"""
        if not self.chunks:
            raise ValueError("Nenhum chunk encontrado. Execute create_chunks() primeiro.")
        
        texts = [chunk.page_content for chunk in self.chunks]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        self.embeddings = embeddings
        return embeddings
    
    def build_faiss_index(self):
        """Constrói índice FAISS para busca semântica rápida"""
        if self.embeddings is None:
            raise ValueError("Embeddings não criados. Execute create_embeddings() primeiro.")
        
        dimension = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner Product (cosine similarity)
        
        # Normalizar embeddings para cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.faiss_index.add(self.embeddings.astype('float32'))
    
    def build_bm25_index(self):
        """Constrói índice BM25 para busca lexical"""
        if not self.chunks:
            raise ValueError("Nenhum chunk encontrado. Execute create_chunks() primeiro.")
        
        texts = [chunk.page_content for chunk in self.chunks]
        tokenized_texts = [text.lower().split() for text in texts]
        self.bm25 = BM25Okapi(tokenized_texts)
    
    def semantic_search(self, query: str, k: int = None) -> List[Dict]:
        """Busca semântica usando FAISS"""
        if self.faiss_index is None:
            raise ValueError("Índice FAISS não construído.")
        
        k = k or self.top_k
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.faiss_index.search(query_embedding.astype('float32'), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                results.append({
                    'content': self.chunks[idx].page_content,
                    'metadata': self.chunks[idx].metadata,
                    'score': float(score),
                    'method': 'semantic'
                })
        
        return results
    
    def lexical_search(self, query: str, k: int = None) -> List[Dict]:
        """Busca lexical usando BM25"""
        if self.bm25 is None:
            raise ValueError("Índice BM25 não construído.")
        
        k = k or self.top_k
        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)
        
        # Obter top-k resultados
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            if idx < len(self.chunks) and scores[idx] > 0:
                results.append({
                    'content': self.chunks[idx].page_content,
                    'metadata': self.chunks[idx].metadata,
                    'score': float(scores[idx]),
                    'method': 'lexical'
                })
        
        return results
    
    def hybrid_search(self, query: str, k: int = None, alpha: float = 0.5) -> List[Dict]:
        """Busca híbrida combinando semântica e lexical"""
        k = k or self.top_k
        
        # Buscar com ambos os métodos
        semantic_results = self.semantic_search(query, k * 2)
        lexical_results = self.lexical_search(query, k * 2)
        
        # Combinar e re-ranquear resultados
        all_results = {}
        
        # Adicionar resultados semânticos
        for result in semantic_results:
            content = result['content']
            if content not in all_results:
                all_results[content] = result.copy()
                all_results[content]['semantic_score'] = result['score']
                all_results[content]['lexical_score'] = 0.0
            else:
                all_results[content]['semantic_score'] = result['score']
        
        # Adicionar resultados lexicais
        for result in lexical_results:
            content = result['content']
            if content not in all_results:
                all_results[content] = result.copy()
                all_results[content]['semantic_score'] = 0.0
                all_results[content]['lexical_score'] = result['score']
            else:
                all_results[content]['lexical_score'] = result['score']
        
        # Normalizar scores e combinar
        semantic_scores = [r['semantic_score'] for r in all_results.values()]
        lexical_scores = [r['lexical_score'] for r in all_results.values()]
        
        if semantic_scores and max(semantic_scores) > 0:
            max_semantic = max(semantic_scores)
            for result in all_results.values():
                result['semantic_score'] /= max_semantic
        
        if lexical_scores and max(lexical_scores) > 0:
            max_lexical = max(lexical_scores)
            for result in all_results.values():
                result['lexical_score'] /= max_lexical
        
        # Combinar scores
        for result in all_results.values():
            result['score'] = (
                alpha * result['semantic_score'] + 
                (1 - alpha) * result['lexical_score']
            )
            result['method'] = 'hybrid'
        
        # Ordenar por score combinado
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x['score'],
            reverse=True
        )
        
        return sorted_results[:k]
    
    def rerank_results(self, query: str, results: List[Dict]) -> List[Dict]:
        """Re-ranqueia resultados usando cross-encoder"""
        # Para simplicidade, usar similaridade de cosseno com query
        query_embedding = self.embedding_model.encode([query])
        
        for result in results:
            content_embedding = self.embedding_model.encode([result['content']])
            similarity = cosine_similarity(query_embedding, content_embedding)[0][0]
            result['rerank_score'] = float(similarity)
        
        # Re-ordenar por rerank_score
        return sorted(results, key=lambda x: x['rerank_score'], reverse=True)
    
    def search(
        self,
        query: str,
        method: str = "hybrid",
        k: int = None,
        rerank: bool = True
    ) -> List[Dict]:
        """Interface principal de busca"""
        k = k or self.top_k
        
        if method == "semantic":
            results = self.semantic_search(query, k)
        elif method == "lexical":
            results = self.lexical_search(query, k)
        elif method == "hybrid":
            results = self.hybrid_search(query, k)
        else:
            raise ValueError(f"Método desconhecido: {method}")
        
        if rerank and results:
            results = self.rerank_results(query, results)
        
        return results
    
    def build_index(self):
        """Constrói todos os índices necessários"""
        print("Carregando documentos...")
        self.load_documents()
        
        print("Criando chunks...")
        self.create_chunks()
        
        print("Criando embeddings...")
        self.create_embeddings()
        
        print("Construindo índice FAISS...")
        self.build_faiss_index()
        
        print("Construindo índice BM25...")
        self.build_bm25_index()
        
        print("Sistema RAG construído com sucesso!")
        print(f"Total de documentos: {len(self.documents)}")
        print(f"Total de chunks: {len(self.chunks)}")
    
    def get_context_for_query(self, query: str, max_context_length: int = 2000) -> str:
        """Obtém contexto relevante para uma query"""
        results = self.search(query, method="hybrid", rerank=True)
        
        context_parts = []
        current_length = 0
        
        for result in results:
            content = result['content']
            if current_length + len(content) <= max_context_length:
                context_parts.append(content)
                current_length += len(content)
            else:
                break
        
        return "\n\n".join(context_parts)


def create_sample_data():
    """Cria dados de exemplo para demonstração"""
    
    # Criar estrutura de pastas
    os.makedirs("data/guias", exist_ok=True)
    os.makedirs("data/emissoes", exist_ok=True)
    os.makedirs("data/avaliacoes", exist_ok=True)
    
    # Guias de viagem sustentáveis
    guia_content = """
    Guia de Viagens Sustentáveis - Brasil
    
    Transporte Sustentável:
    - Prefira trens para viagens de longa distância quando disponível
    - Use ônibus para trajetos regionais - menor emissão de CO2 por passageiro
    - Evite voos domésticos quando possível
    - Para voos internacionais, prefira voos diretos
    
    São Paulo - Rio de Janeiro:
    - Trem: Não disponível atualmente
    - Ônibus: 6h de viagem, baixa emissão de carbono
    - Avião: 1h30, alta emissão de carbono
    - Carro: 6h, emissão moderada se compartilhado
    
    Hospedagem Sustentável:
    - Hotéis com certificação LEED ou similar
    - Pousadas locais que empregam comunidade
    - Hostels com práticas ecológicas
    
    Atividades Sustentáveis:
    - Turismo de base comunitária
    - Ecoturismo em áreas protegidas
    - Visitas a projetos de conservação
    - Compras em mercados locais
    """
    
    with open("data/guias/guia_sustentavel_brasil.txt", "w", encoding="utf-8") as f:
        f.write(guia_content)
    
    # Dados de emissões
    emissoes_data = {
        "modal_transporte": ["aviao", "carro", "onibus", "trem", "moto"],
        "emissao_co2_kg_km": [0.255, 0.171, 0.089, 0.041, 0.113],
        "capacidade_passageiros": [150, 4, 50, 200, 2],
        "custo_medio_km": [0.35, 0.25, 0.08, 0.12, 0.15]
    }
    
    df_emissoes = pd.DataFrame(emissoes_data)
    df_emissoes.to_csv("data/emissoes/emissoes_transporte.csv", index=False)
    
    # Avaliações de hotéis
    avaliacoes_data = {
        "hotéis_sustentáveis": [
            {
                "nome": "Eco Resort Itacaré",
                "cidade": "Itacaré",
                "estado": "BA",
                "certificacao": "LEED Gold",
                "nota_sustentabilidade": 9.2,
                "práticas": ["energia solar", "reciclagem", "água da chuva"],
                "preço_diária": 280
            },
            {
                "nome": "Pousada Verde Rio",
                "cidade": "Rio de Janeiro", 
                "estado": "RJ",
                "certificacao": "Green Key",
                "nota_sustentabilidade": 8.5,
                "práticas": ["energia renovável", "produtos locais"],
                "preço_diária": 180
            }
        ]
    }
    
    with open("data/avaliacoes/hoteis_sustentaveis.json", "w", encoding="utf-8") as f:
        json.dump(avaliacoes_data, f, indent=2, ensure_ascii=False)
    
    print("Dados de exemplo criados com sucesso!")


if __name__ == "__main__":
    # Criar dados de exemplo
    create_sample_data()
    
    # Testar sistema RAG
    rag = AdvancedRAGSystem()
    rag.build_index()
    
    # Teste de busca
    query = "Como viajar de São Paulo para Rio de Janeiro de forma sustentável?"
    results = rag.search(query)
    
    print(f"\nResultados para: {query}")
    for i, result in enumerate(results[:3]):
        print(f"\n{i+1}. Score: {result['score']:.3f}")
        print(f"Método: {result['method']}")
        print(f"Conteúdo: {result['content'][:200]}...")