import os
import logging
from typing import List
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models.schemas import QueryRequest, QueryResponse
from services.document_processor import DocumentProcessor
from services.simple_embeddings import SimpleEmbeddingService as EmbeddingService
from services.simple_vector_store import SimpleVectorStore as VectorStore
from services.gemini_service import GeminiService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Legal RAG Backend",
    description="Retrieval-Augmented Generation service for legal document queries",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize services
document_processor = DocumentProcessor()
embedding_service = EmbeddingService()
vector_store = VectorStore()
gemini_service = GeminiService()

# Global variable to track initialization
_initialized = False

async def initialize_services():
    """Initialize all services and process documents"""
    global _initialized
    if _initialized:
        return
    
    try:
        logger.info("Initializing services...")
        
        # Initialize embedding service
        await embedding_service.initialize()
        
        # Process documents if any exist
        documents_dir = "documents"
        if os.path.exists(documents_dir) and os.listdir(documents_dir):
            logger.info("Processing documents...")
            documents = document_processor.process_documents_from_directory(documents_dir)
            
            if documents:
                # Create embeddings
                logger.info("Creating embeddings...")
                embeddings = await embedding_service.create_embeddings([doc.content for doc in documents])
                
                # Store in vector database
                logger.info("Storing in vector database...")
                vector_store.add_documents(documents, embeddings)
                
                logger.info(f"Successfully processed {len(documents)} document chunks")
            else:
                logger.warning("No documents found to process")
        else:
            logger.warning("Documents directory is empty or doesn't exist")
        
        _initialized = True
        logger.info("Services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    await initialize_services()

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML page"""
    return FileResponse("static/index.html")

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query legal documents using RAG approach
    
    Args:
        request: QueryRequest containing the natural language query
        
    Returns:
        QueryResponse with generated answer and citations
    """
    try:
        # Ensure services are initialized
        await initialize_services()
        
        if not request.query or not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        logger.info(f"Processing query: {request.query}")
        
        # Check if vector store has documents
        if vector_store.is_empty():
            raise HTTPException(
                status_code=400, 
                detail="No documents have been processed. Please add documents to the 'documents' directory and restart the service."
            )
        
        # Create query embedding
        query_embedding = await embedding_service.create_embeddings([request.query])
        
        # Search for relevant documents
        relevant_docs = vector_store.search(query_embedding[0], top_k=5)
        
        if not relevant_docs:
            return QueryResponse(
                answer="I couldn't find relevant information in the available documents to answer your query.",
                citations=[]
            )
        
        # Generate answer using Gemini
        answer, citations = await gemini_service.generate_answer_with_citations(
            query=request.query,
            relevant_docs=relevant_docs
        )
        
        return QueryResponse(answer=answer, citations=citations)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "initialized": _initialized,
        "documents_count": vector_store.get_document_count() if _initialized else 0
    }

@app.get("/documents/count")
async def get_document_count():
    """Get the number of processed documents"""
    await initialize_services()
    return {"count": vector_store.get_document_count()}


