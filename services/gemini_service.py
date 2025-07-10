import os
import logging
from typing import List, Tuple
from google import genai
from google.genai import types
from services.document_processor import Document
from models.schemas import Citation

logger = logging.getLogger(__name__)

class GeminiService:
    """Service for generating answers using Gemini API"""
    
    def __init__(self):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", "AIzaSyAbcupSfRthfVPj07cijh4mI4bwAcgYCaA"))
        self.model_name = "gemini-2.5-flash"
    
    async def generate_answer_with_citations(self, query: str, relevant_docs: List[Document]) -> Tuple[str, List[Citation]]:
        """
        Generate an answer with citations based on relevant documents
        
        Args:
            query: User's query
            relevant_docs: List of relevant documents
            
        Returns:
            Tuple of (answer, citations)
        """
        try:
            # Prepare context from relevant documents
            context = self._prepare_context(relevant_docs)
            
            # Generate answer
            answer = await self._generate_answer(query, context)
            
            # Extract citations from relevant documents
            citations = self._extract_citations(relevant_docs, answer)
            
            return answer, citations
            
        except Exception as e:
            logger.error(f"Failed to generate answer with citations: {e}")
            raise
    
    def _prepare_context(self, documents: List[Document]) -> str:
        """Prepare context from relevant documents"""
        context_parts = []
        
        for i, doc in enumerate(documents):
            context_parts.append(f"Document {i+1} (Source: {doc.source}):\n{doc.content}\n")
        
        return "\n".join(context_parts)
    
    async def _generate_answer(self, query: str, context: str) -> str:
        """Generate answer using Gemini"""
        try:
            system_prompt = """You are a legal expert assistant. Based on the provided legal documents, answer the user's query accurately and comprehensively. 

Important guidelines:
1. Base your answer strictly on the provided documents
2. If information is not available in the documents, clearly state this
3. Provide a clear, well-structured response
4. Use formal legal language when appropriate
5. Be precise and factual
6. If citing specific legal principles or cases, reference them clearly

Format your response as a comprehensive answer without any special formatting for citations - citations will be handled separately."""

            prompt = f"""Query: {query}

Legal Documents:
{context}

Please provide a comprehensive answer based on the legal documents provided above."""

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Content(role="user", parts=[types.Part(text=prompt)])
                ],
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.3,
                    max_output_tokens=1000
                )
            )
            
            if response.text:
                return response.text.strip()
            else:
                return "I apologize, but I couldn't generate a response based on the available documents."
                
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            return "I encountered an error while processing your query. Please try again."
    
    def _extract_citations(self, documents: List[Document], answer: str) -> List[Citation]:
        """Extract relevant citations from documents"""
        citations = []
        
        try:
            for doc in documents:
                # Extract relevant sentences from the document
                sentences = self._extract_relevant_sentences(doc.content)
                
                for sentence in sentences:
                    if len(sentence.strip()) > 20:  # Only include substantial sentences
                        citation = Citation(
                            text=sentence.strip(),
                            source=doc.source
                        )
                        citations.append(citation)
                        
                        # Limit citations to avoid overwhelming response
                        if len(citations) >= 6:
                            break
                
                if len(citations) >= 6:
                    break
            
            # Remove duplicates while preserving order
            unique_citations = []
            seen_texts = set()
            
            for citation in citations:
                if citation.text not in seen_texts:
                    unique_citations.append(citation)
                    seen_texts.add(citation.text)
            
            return unique_citations[:4]  # Return top 4 unique citations
            
        except Exception as e:
            logger.error(f"Failed to extract citations: {e}")
            return []
    
    def _extract_relevant_sentences(self, text: str) -> List[str]:
        """Extract relevant sentences from document text"""
        # Split text into sentences
        sentences = []
        
        # Simple sentence splitting
        import re
        sentence_endings = re.split(r'[.!?]+', text)
        
        for sentence in sentence_endings:
            sentence = sentence.strip()
            if len(sentence) > 30:  # Only substantial sentences
                # Clean up the sentence
                sentence = re.sub(r'\s+', ' ', sentence)
                sentences.append(sentence)
        
        # Return first few sentences that are substantial
        return sentences[:3]
