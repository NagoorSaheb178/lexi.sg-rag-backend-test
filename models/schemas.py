from typing import List, Optional
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    """Request model for document queries"""
    query: str = Field(..., description="Natural language query about legal documents", min_length=1)

class Citation(BaseModel):
    """Model for document citations"""
    text: str = Field(..., description="Relevant text excerpt from the document")
    source: str = Field(..., description="Source document name")

class QueryResponse(BaseModel):
    """Response model for document queries"""
    answer: str = Field(..., description="Generated answer based on document retrieval")
    citations: List[Citation] = Field(..., description="List of citations supporting the answer")

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    initialized: bool
    documents_count: int
