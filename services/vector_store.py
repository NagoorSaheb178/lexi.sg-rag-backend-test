import os
import json
import logging
from typing import List, Tuple, Optional
import numpy as np
import faiss
from services.document_processor import Document

logger = logging.getLogger(__name__)

class VectorStore:
    """FAISS-based vector store for document retrieval"""
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.index = None
        self.documents = []
        self.metadata_file = "data/processed_documents.json"
        
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        
        # Initialize FAISS index
        self._initialize_index()
        
        # Load existing data if available
        self._load_existing_data()
    
    def _initialize_index(self):
        """Initialize FAISS index"""
        try:
            # Use IndexFlatIP for cosine similarity
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            logger.info(f"Initialized FAISS index with dimension {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to initialize FAISS index: {e}")
            raise
    
    def _load_existing_data(self):
        """Load existing documents and index if available"""
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
                    embeddings = np.array(embeddings_data, dtype=np.float32)
                    # Normalize embeddings for cosine similarity
                    faiss.normalize_L2(embeddings)
                    self.index.add(embeddings)
                    logger.info(f"Loaded {len(self.documents)} documents from cache")
                else:
                    logger.warning("Embeddings data inconsistent, clearing cache")
                    self.documents = []
                    self._initialize_index()
            
        except Exception as e:
            logger.error(f"Failed to load existing data: {e}")
            self.documents = []
            self._initialize_index()
    
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
            # Clear existing data
            self.documents = []
            self._initialize_index()
            
            # Add new documents
            self.documents.extend(documents)
            
            # Convert embeddings to numpy array and normalize
            embeddings_array = np.array(embeddings, dtype=np.float32)
            faiss.normalize_L2(embeddings_array)
            
            # Add to FAISS index
            self.index.add(embeddings_array)
            
            # Save to cache
            self._save_data(embeddings_array)
            
            logger.info(f"Added {len(documents)} documents to vector store")
            
        except Exception as e:
            logger.error(f"Failed to add documents to vector store: {e}")
            raise
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Document]:
        """
        Search for similar documents
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            
        Returns:
            List of most similar documents
        """
        if self.index.ntotal == 0:
            logger.warning("Vector store is empty")
            return []
        
        try:
            # Normalize query embedding
            query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
            faiss.normalize_L2(query_embedding)
            
            # Search
            similarities, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
            
            # Return documents sorted by similarity
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.documents):
                    results.append(self.documents[idx])
            
            logger.info(f"Found {len(results)} similar documents")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search vector store: {e}")
            return []
    
    def _save_data(self, embeddings: np.ndarray):
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
                'embeddings': embeddings.tolist()
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
