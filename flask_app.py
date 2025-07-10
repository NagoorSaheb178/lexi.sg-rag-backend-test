"""
Flask wrapper for Legal RAG Backend
This provides WSGI compatibility for the FastAPI application
"""
import os
import json
import logging
import asyncio
from flask import Flask, request, jsonify, render_template_string
from flask import send_from_directory
from werkzeug.exceptions import HTTPException

# Import the FastAPI services directly
from services.document_processor import DocumentProcessor
from services.simple_embeddings import SimpleEmbeddingService
from services.simple_vector_store import SimpleVectorStore
from services.gemini_service import GeminiService
from models.schemas import QueryRequest, QueryResponse, Citation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

# Initialize services
document_processor = DocumentProcessor()
embedding_service = SimpleEmbeddingService()
vector_store = SimpleVectorStore()
gemini_service = GeminiService()

# Global initialization flag
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

def run_async(coro):
    """Run async function in sync context"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

@app.route('/')
def index():
    """Serve the main HTML page"""
    try:
        with open('static/index.html', 'r') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error serving index.html: {e}")
        return "Error loading index page", 500

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

@app.route('/health')
def health_check():
    """Health check endpoint"""
    try:
        return jsonify({
            "status": "healthy",
            "initialized": _initialized,
            "documents_count": vector_store.get_document_count() if _initialized else 0
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/documents/count')
def get_document_count():
    """Get the number of processed documents"""
    try:
        if not _initialized:
            run_async(initialize_services())
        return jsonify({"count": vector_store.get_document_count()})
    except Exception as e:
        logger.error(f"Error getting document count: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/query', methods=['POST'])
def query_documents():
    """Query legal documents using RAG approach"""
    try:
        # Initialize services if not already done
        if not _initialized:
            run_async(initialize_services())
        
        # Get request data
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "Query is required"}), 400
        
        query = data['query'].strip()
        if not query:
            return jsonify({"error": "Query cannot be empty"}), 400
        
        logger.info(f"Processing query: {query}")
        
        # Check if vector store has documents
        if vector_store.is_empty():
            return jsonify({
                "error": "No documents have been processed. Please add documents to the 'documents' directory and restart the service."
            }), 400
        
        # Create query embedding
        query_embedding = run_async(embedding_service.create_embeddings([query]))
        
        # Search for relevant documents
        relevant_docs = vector_store.search(query_embedding[0], top_k=5)
        
        if not relevant_docs:
            return jsonify({
                "answer": "I couldn't find relevant information in the available documents to answer your query.",
                "citations": []
            })
        
        # Generate answer using Gemini
        answer, citations = run_async(gemini_service.generate_answer_with_citations(
            query=query,
            relevant_docs=relevant_docs
        ))
        
        # Format response
        response = {
            "answer": answer,
            "citations": [{"text": c.text, "source": c.source} for c in citations]
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    """Handle all exceptions"""
    logger.error(f"Unhandled exception: {e}")
    if isinstance(e, HTTPException):
        return jsonify({"error": e.description}), e.code
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)