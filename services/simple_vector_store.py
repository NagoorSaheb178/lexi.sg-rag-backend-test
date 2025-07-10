import os
import json
import logging
from typing import List, Tuple, Optional
import numpy as np
from services.document_processor import Document

logger = logging.getLogger(__name__)

class SimpleVectorStore:
    """Simple vector store using cosine similarity (fallback for FAISS)"""
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.documents = []
        self.embeddings = []
        self.metadata_file = "data/processed_documents.json"
        
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        
        # Load existing data if available
        self._load_existing_data()
    
    def _load_existing_data(self):
        """Load existing documents and embeddings if available"""
        try:
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                    
                # Reconstruct documents
                self.documents = []
                for doc_data in data.get('documents', []):
                    doc = Document(
                        content=doc_data['content'],
                        source=doc_data['source'],
                        chunk_id=doc_data['chunk_id'],
                        metadata=doc_data['metadata']
                    )
                    self.documents.append(doc)
                
                # Reload embeddings if available
                embeddings_data = data.get('embeddings', [])
                if embeddings_data and len(embeddings_data) == len(self.documents):
                    self.embeddings = [np.array(emb, dtype=np.float32) for emb in embeddings_data]
                    logger.info(f"Loaded {len(self.documents)} documents from cache")
                else:
                    logger.warning("Embeddings data inconsistent, clearing cache")
                    self.documents = []
                    self.embeddings = []
            
        except Exception as e:
            logger.error(f"Failed to load existing data: {e}")
            self.documents = []
            self.embeddings = []
    
    def add_documents(self, documents: List[Document], embeddings: List[np.ndarray]):
        """
        Add documents and their embeddings to the vector store
        
        Args:
            documents: List of Document objects
            embeddings: List of embedding vectors
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        
        try:
            # Clear existing data and add new
            self.documents = documents.copy()
            self.embeddings = [emb.astype(np.float32) for emb in embeddings]
            
            # Save to cache
            self._save_data()
            
            logger.info(f"Added {len(documents)} documents to vector store")
            
        except Exception as e:
            logger.error(f"Failed to add documents to vector store: {e}")
            raise
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Document]:
        """
        Search for similar documents using cosine similarity
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            
        Returns:
            List of most similar documents
        """
        if not self.embeddings:
            logger.warning("Vector store is empty")
            return []
        
        try:
            # Normalize query embedding
            query_embedding = query_embedding.astype(np.float32)
            query_norm = np.linalg.norm(query_embedding)
            if query_norm > 0:
                query_embedding = query_embedding / query_norm
            
            # Calculate cosine similarities
            similarities = []
            for i, doc_embedding in enumerate(self.embeddings):
                # Normalize document embedding
                doc_norm = np.linalg.norm(doc_embedding)
                if doc_norm > 0:
                    normalized_doc = doc_embedding / doc_norm
                    similarity = np.dot(query_embedding, normalized_doc)
                else:
                    similarity = 0.0
                similarities.append((similarity, i))
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[0], reverse=True)
            
            # Return top_k documents
            results = []
            for similarity, idx in similarities[:top_k]:
                if idx < len(self.documents):
                    results.append(self.documents[idx])
            
            logger.info(f"Found {len(results)} similar documents")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search vector store: {e}")
            return []
    
    def _save_data(self):
        """Save documents and embeddings to cache"""
        try:
            data = {
                'documents': [
                    {
                        'content': doc.content,
                        'source': doc.source,
                        'chunk_id': doc.chunk_id,
                        'metadata': doc.metadata
                    }
                    for doc in self.documents
                ],
                'embeddings': [emb.tolist() for emb in self.embeddings]
            }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved {len(self.documents)} documents to cache")
            
        except Exception as e:
            logger.error(f"Failed to save data to cache: {e}")
    
    def is_empty(self) -> bool:
        """Check if vector store is empty"""
        return len(self.documents) == 0
    
    def get_document_count(self) -> int:
        """Get the number of documents in the store"""
        return len(self.documents)