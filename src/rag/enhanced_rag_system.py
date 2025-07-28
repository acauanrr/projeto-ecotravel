"""
Sistema RAG Aprimorado para o EcoTravel Agent
Implementa busca híbrida (BM25 + Semantic), reranking e anti-alucinação
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
from rank_bm25 import BM25Okapi

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

try:
    # Tentar imports novos primeiro
    from langchain_community.document_loaders import DirectoryLoader, TextLoader
except ImportError:
    # Fallback para imports antigos
    from langchain.document_loaders import DirectoryLoader, TextLoader
import torch

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedRAGSystem:
    """Sistema RAG aprimorado com busca híbrida e reranking"""
    
    def __init__(
        self,
        data_path: str = "data",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        chunk_size: int = 512,
        chunk_overlap: int = 100,
        top_k_retrieve: int = 20,
        top_k_rerank: int = 5,
        bm25_weight: float = 0.5
    ):
        """
        Inicializa o sistema RAG aprimorado
        
        Args:
            data_path: Caminho para os dados
            embedding_model: Modelo para embeddings semânticos
            reranker_model: Modelo para reranking
            chunk_size: Tamanho dos chunks
            chunk_overlap: Sobreposição entre chunks
            top_k_retrieve: Número de documentos para recuperar
            top_k_rerank: Número final após reranking
            bm25_weight: Peso para busca BM25 (0-1)
        """
        self.data_path = Path(data_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k_retrieve = top_k_retrieve
        self.top_k_rerank = top_k_rerank
        self.bm25_weight = bm25_weight
        
        # Inicializar modelos
        logger.info(f"Carregando modelo de embeddings: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        logger.info(f"Carregando modelo de reranking: {reranker_model}")
        self.reranker = CrossEncoder(reranker_model)
        
        # Text splitter para chunking inteligente
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "],
            length_function=len
        )
        
        # Armazenamento
        self.documents = []
        self.chunks = []
        self.chunk_metadata = []
        self.embeddings = None
        self.bm25 = None
        self.index = None
        
        # Estatísticas
        self.stats = {
            "total_documents": 0,
            "total_chunks": 0,
            "avg_chunk_size": 0,
            "sources": []
        }
        
    def load_data(self) -> None:
        """Carrega e processa todos os dados disponíveis"""
        logger.info("Carregando dados...")
        
        # Carregar guias de turismo
        guias_path = self.data_path / "guias"
        if guias_path.exists():
            self._load_text_files(guias_path, "guia_turismo")
            
        # Carregar dados de emissões
        emissoes_path = self.data_path / "emissoes"
        if emissoes_path.exists():
            self._load_text_files(emissoes_path, "emissoes_co2")
            
        # Carregar avaliações de hotéis
        avaliacoes_path = self.data_path / "avaliacoes"
        if avaliacoes_path.exists():
            self._load_text_files(avaliacoes_path, "avaliacoes_hotel")
            
        # Carregar destinos
        destinos_path = self.data_path / "destinos"
        if destinos_path.exists():
            self._load_text_files(destinos_path, "destinos_sustentaveis")
            
        logger.info(f"Total de documentos carregados: {len(self.documents)}")
        
        # Criar chunks
        self._create_chunks()
        
        # Criar índices
        self._create_indices()
        
        # Atualizar estatísticas
        self._update_stats()
        
    def _load_text_files(self, directory: Path, source_type: str) -> None:
        """Carrega arquivos de texto de um diretório"""
        for file_path in directory.glob("*.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": str(file_path),
                        "source_type": source_type,
                        "filename": file_path.name,
                        "load_timestamp": datetime.now().isoformat()
                    }
                )
                self.documents.append(doc)
                logger.info(f"Carregado: {file_path.name}")
                
            except Exception as e:
                logger.error(f"Erro ao carregar {file_path}: {e}")
                
    def _create_chunks(self) -> None:
        """Cria chunks dos documentos com metadados enriquecidos"""
        logger.info("Criando chunks dos documentos...")
        
        for doc in self.documents:
            # Dividir documento em chunks
            doc_chunks = self.text_splitter.split_text(doc.page_content)
            
            for i, chunk_text in enumerate(doc_chunks):
                # Criar metadados enriquecidos
                chunk_metadata = {
                    **doc.metadata,
                    "chunk_index": i,
                    "chunk_total": len(doc_chunks),
                    "chunk_size": len(chunk_text),
                    "has_numbers": any(char.isdigit() for char in chunk_text),
                    "has_percentages": "%" in chunk_text,
                    "has_co2_mention": any(term in chunk_text.lower() for term in ["co2", "carbono", "emissão", "emissões"])
                }
                
                self.chunks.append(chunk_text)
                self.chunk_metadata.append(chunk_metadata)
                
        logger.info(f"Total de chunks criados: {len(self.chunks)}")
        
    def _create_indices(self) -> None:
        """Cria índices para busca híbrida"""
        logger.info("Criando índices de busca...")
        
        # Criar embeddings semânticos
        logger.info("Gerando embeddings...")
        self.embeddings = self.embedding_model.encode(
            self.chunks,
            show_progress_bar=True,
            batch_size=32
        )
        
        # Criar índice FAISS
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product para similaridade
        
        # Normalizar embeddings para usar inner product como cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
        
        # Criar índice BM25
        logger.info("Criando índice BM25...")
        tokenized_chunks = [chunk.lower().split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)
        
    def search(
        self,
        query: str,
        filter_source_type: Optional[str] = None,
        use_reranking: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Busca híbrida com reranking opcional
        
        Args:
            query: Query de busca
            filter_source_type: Filtrar por tipo de fonte
            use_reranking: Se deve usar reranking
            
        Returns:
            Lista de resultados rankeados
        """
        # Busca semântica
        semantic_results = self._semantic_search(query)
        
        # Busca BM25
        bm25_results = self._bm25_search(query)
        
        # Combinar resultados
        combined_results = self._combine_results(semantic_results, bm25_results)
        
        # Filtrar por tipo de fonte se especificado
        if filter_source_type:
            combined_results = [
                r for r in combined_results 
                if r["metadata"]["source_type"] == filter_source_type
            ]
        
        # Reranking se habilitado
        if use_reranking and len(combined_results) > 0:
            combined_results = self._rerank_results(query, combined_results)
            
        # Retornar top-k final
        return combined_results[:self.top_k_rerank]
        
    def _semantic_search(self, query: str) -> List[Dict[str, Any]]:
        """Busca semântica usando embeddings"""
        # Gerar embedding da query
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Buscar no índice
        distances, indices = self.index.search(query_embedding, self.top_k_retrieve)
        
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx != -1:  # FAISS retorna -1 para resultados não encontrados
                results.append({
                    "chunk": self.chunks[idx],
                    "metadata": self.chunk_metadata[idx],
                    "score": float(dist),
                    "rank": i,
                    "method": "semantic"
                })
                
        return results
        
    def _bm25_search(self, query: str) -> List[Dict[str, Any]]:
        """Busca BM25 (lexical)"""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Obter top-k índices
        top_indices = np.argsort(scores)[::-1][:self.top_k_retrieve]
        
        results = []
        for i, idx in enumerate(top_indices):
            if scores[idx] > 0:  # Apenas resultados com score > 0
                results.append({
                    "chunk": self.chunks[idx],
                    "metadata": self.chunk_metadata[idx],
                    "score": float(scores[idx]),
                    "rank": i,
                    "method": "bm25"
                })
                
        return results
        
    def _combine_results(
        self,
        semantic_results: List[Dict],
        bm25_results: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Combina resultados de busca semântica e BM25"""
        # Normalizar scores
        semantic_scores = {r["chunk"]: r["score"] for r in semantic_results}
        bm25_scores = {r["chunk"]: r["score"] for r in bm25_results}
        
        # Obter todos os chunks únicos
        all_chunks = set(semantic_scores.keys()) | set(bm25_scores.keys())
        
        combined = []
        for chunk in all_chunks:
            # Calcular score combinado
            sem_score = semantic_scores.get(chunk, 0)
            bm25_score = bm25_scores.get(chunk, 0)
            
            # Normalizar scores para [0, 1]
            max_sem = max(semantic_scores.values()) if semantic_scores else 1
            max_bm25 = max(bm25_scores.values()) if bm25_scores else 1
            
            norm_sem = sem_score / max_sem if max_sem > 0 else 0
            norm_bm25 = bm25_score / max_bm25 if max_bm25 > 0 else 0
            
            # Score híbrido ponderado
            hybrid_score = (1 - self.bm25_weight) * norm_sem + self.bm25_weight * norm_bm25
            
            # Encontrar metadados
            for r in semantic_results + bm25_results:
                if r["chunk"] == chunk:
                    combined.append({
                        "chunk": chunk,
                        "metadata": r["metadata"],
                        "score": hybrid_score,
                        "semantic_score": norm_sem,
                        "bm25_score": norm_bm25,
                        "method": "hybrid"
                    })
                    break
                    
        # Ordenar por score híbrido
        combined.sort(key=lambda x: x["score"], reverse=True)
        
        return combined
        
    def _rerank_results(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Rerankeia resultados usando cross-encoder"""
        if not results:
            return results
            
        # Preparar pares query-documento
        pairs = [[query, r["chunk"]] for r in results]
        
        # Calcular scores de reranking
        rerank_scores = self.reranker.predict(pairs)
        
        # Adicionar scores e reordenar
        for i, (result, score) in enumerate(zip(results, rerank_scores)):
            result["rerank_score"] = float(score)
            
        # Ordenar por score de reranking
        results.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        return results
        
    def _update_stats(self) -> None:
        """Atualiza estatísticas do sistema"""
        self.stats["total_documents"] = len(self.documents)
        self.stats["total_chunks"] = len(self.chunks)
        self.stats["avg_chunk_size"] = np.mean([len(c) for c in self.chunks]) if self.chunks else 0
        self.stats["sources"] = list(set(m["source_type"] for m in self.chunk_metadata))
        
    def get_context_for_query(
        self,
        query: str,
        max_tokens: int = 2000
    ) -> Tuple[str, List[Dict[str, str]]]:
        """
        Obtém contexto otimizado para uma query
        
        Args:
            query: Query do usuário
            max_tokens: Máximo de tokens no contexto
            
        Returns:
            Contexto formatado e lista de fontes
        """
        try:
            results = self.search(query, use_reranking=True)
            
            if not results:
                return "Não foram encontradas informações relevantes nos documentos disponíveis.", []
                
            # Construir contexto
            context_parts = []
            sources = []
            current_tokens = 0
            
            for result in results:
                chunk_tokens = len(result["chunk"].split())
                
                if current_tokens + chunk_tokens > max_tokens:
                    break
                    
                context_parts.append(f"[Fonte: {result['metadata']['filename']}]\n{result['chunk']}")
                sources.append({
                    "filename": result['metadata']['filename'],
                    "source_type": result['metadata']['source_type'],
                    "score": result.get('score', 0)
                })
                current_tokens += chunk_tokens
                
            if not context_parts:
                return "Não foram encontradas informações relevantes nos documentos disponíveis.", []
                
            context = "\n\n---\n\n".join(context_parts)
            
            return context, sources
            
        except Exception as e:
            logger.error(f"Erro em get_context_for_query: {e}")
            # Fallback: tentar busca simples
            try:
                results = self.search(query, use_reranking=False)
                if results:
                    best_result = results[0]
                    return best_result['chunk'], [{
                        "filename": best_result['metadata']['filename'],
                        "source_type": best_result['metadata']['source_type'],
                        "score": best_result.get('score', 0)
                    }]
                else:
                    return "Sistema RAG temporariamente indisponível.", []
            except Exception as e2:
                logger.error(f"Erro no fallback: {e2}")
                return "Sistema RAG temporariamente indisponível.", []
        
    def save_index(self, path: str) -> None:
        """Salva índices para uso posterior"""
        save_path = Path(path)
        save_path.mkdir(exist_ok=True)
        
        # Salvar índice FAISS
        faiss.write_index(self.index, str(save_path / "faiss.index"))
        
        # Salvar embeddings e metadados
        np.save(save_path / "embeddings.npy", self.embeddings)
        
        with open(save_path / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump({
                "chunks": self.chunks,
                "metadata": self.chunk_metadata,
                "stats": self.stats
            }, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Índices salvos em: {save_path}")
        
    def load_index(self, path: str) -> None:
        """Carrega índices salvos"""
        load_path = Path(path)
        
        # Carregar índice FAISS
        self.index = faiss.read_index(str(load_path / "faiss.index"))
        
        # Carregar embeddings e metadados
        self.embeddings = np.load(load_path / "embeddings.npy")
        
        with open(load_path / "metadata.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.chunks = data["chunks"]
            self.chunk_metadata = data["metadata"]
            self.stats = data["stats"]
            
        # Recriar índice BM25
        tokenized_chunks = [chunk.lower().split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)
        
        logger.info(f"Índices carregados de: {load_path}")


# Função auxiliar para teste rápido
def test_enhanced_rag():
    """Testa o sistema RAG aprimorado"""
    rag = EnhancedRAGSystem()
    rag.load_data()
    
    # Queries de teste
    test_queries = [
        "Como reduzir emissões de CO2 em viagens?",
        "Quais são os melhores destinos sustentáveis no Brasil?",
        "Hotéis eco-friendly no Nordeste",
        "Transporte sustentável entre cidades"
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print(f"{'='*50}")
        
        results = rag.search(query)
        
        for i, result in enumerate(results[:3]):
            print(f"\n--- Resultado {i+1} ---")
            print(f"Fonte: {result['metadata']['filename']}")
            print(f"Score: {result['score']:.3f}")
            if 'rerank_score' in result:
                print(f"Rerank Score: {result['rerank_score']:.3f}")
            print(f"Trecho: {result['chunk'][:200]}...")
            

if __name__ == "__main__":
    test_enhanced_rag() 