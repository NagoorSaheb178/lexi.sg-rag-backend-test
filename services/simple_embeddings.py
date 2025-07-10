import logging
from typing import List
import numpy as np
from collections import Counter
import re

logger = logging.getLogger(__name__)

class SimpleEmbeddingService:
    """Simple TF-IDF based embedding service as fallback"""
    
    def __init__(self):
        self.vocabulary = {}
        self.idf_values = {}
        self.embedding_dim = 384  # Match expected dimension
        self.initialized = False
    
    async def initialize(self):
        """Initialize the embedding service"""
        self.initialized = True
        logger.info("Simple embedding service initialized")
    
    async def create_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """
        Create simple TF-IDF based embeddings
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        if not self.initialized:
            await self.initialize()
        
        # Build vocabulary if not already built
        if not self.vocabulary:
            self._build_vocabulary(texts)
        
        embeddings = []
        for text in texts:
            embedding = self._text_to_embedding(text)
            embeddings.append(embedding)
        
        logger.info(f"Created {len(embeddings)} simple embeddings")
        return embeddings
    
    def _build_vocabulary(self, texts: List[str]):
        """Build vocabulary from texts"""
        all_words = []
        for text in texts:
            words = self._tokenize(text)
            all_words.extend(words)
        
        # Create vocabulary with most common words
        word_counts = Counter(all_words)
        most_common = word_counts.most_common(min(1000, len(word_counts)))
        
        self.vocabulary = {word: i for i, (word, _) in enumerate(most_common)}
        
        # Calculate IDF values
        doc_count = len(texts)
        word_doc_count = Counter()
        
        for text in texts:
            words = set(self._tokenize(text))
            for word in words:
                word_doc_count[word] += 1
        
        for word in self.vocabulary:
            self.idf_values[word] = np.log(doc_count / (word_doc_count.get(word, 1) + 1))
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        # Convert to lowercase and split on non-alphanumeric
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        return words
    
    def _text_to_embedding(self, text: str) -> np.ndarray:
        """Convert text to embedding vector"""
        words = self._tokenize(text)
        word_counts = Counter(words)
        
        # Create sparse vector
        embedding = np.zeros(self.embedding_dim)
        
        for word, count in word_counts.items():
            if word in self.vocabulary:
                idx = self.vocabulary[word] % self.embedding_dim
                tf = count / len(words) if words else 0
                idf = self.idf_values.get(word, 1.0)
                embedding[idx] += tf * idf
        
        # Add some randomness to fill the embedding space
        if np.sum(embedding) > 0:
            # Normalize
            embedding = embedding / np.linalg.norm(embedding)
        else:
            # Random embedding for unknown texts
            embedding = np.random.normal(0, 0.1, self.embedding_dim)
            embedding = embedding / np.linalg.norm(embedding)
        
        return embedding.astype(np.float32)
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors"""
        return self.embedding_dim