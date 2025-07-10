#!/usr/bin/env python3
"""
Test script for Legal RAG Backend
"""
import asyncio
import json
from app import app, initialize_services

async def test_initialization():
    """Test if services initialize correctly"""
    print("Testing service initialization...")
    try:
        await initialize_services()
        print("✓ Services initialized successfully")
        return True
    except Exception as e:
        print(f"✗ Service initialization failed: {e}")
        return False

async def test_document_processing():
    """Test document processing"""
    print("\nTesting document processing...")
    from services.document_processor import DocumentProcessor
    
    try:
        processor = DocumentProcessor()
        documents = processor.process_documents_from_directory("documents")
        print(f"✓ Processed {len(documents)} document chunks")
        
        if documents:
            print(f"  - First document: {documents[0].source}")
            print(f"  - Content preview: {documents[0].content[:100]}...")
        
        return len(documents) > 0
    except Exception as e:
        print(f"✗ Document processing failed: {e}")
        return False

async def test_embeddings():
    """Test embedding generation"""
    print("\nTesting embeddings...")
    from services.simple_embeddings import SimpleEmbeddingService
    
    try:
        embedding_service = SimpleEmbeddingService()
        await embedding_service.initialize()
        
        test_texts = ["This is a test document", "Another test text"]
        embeddings = await embedding_service.create_embeddings(test_texts)
        
        print(f"✓ Generated {len(embeddings)} embeddings")
        print(f"  - Embedding dimension: {len(embeddings[0])}")
        
        return True
    except Exception as e:
        print(f"✗ Embedding generation failed: {e}")
        return False

async def test_vector_store():
    """Test vector store operations"""
    print("\nTesting vector store...")
    from services.simple_vector_store import SimpleVectorStore
    from services.document_processor import Document
    import numpy as np
    
    try:
        vector_store = SimpleVectorStore()
        
        # Create test documents
        test_docs = [
            Document("Insurance law content", "test1.txt", "chunk1", {}),
            Document("Vehicle permit regulations", "test2.txt", "chunk2", {})
        ]
        
        # Create test embeddings
        test_embeddings = [
            np.random.rand(384).astype(np.float32),
            np.random.rand(384).astype(np.float32)
        ]
        
        vector_store.add_documents(test_docs, test_embeddings)
        
        # Test search
        query_embedding = np.random.rand(384).astype(np.float32)
        results = vector_store.search(query_embedding, top_k=2)
        
        print(f"✓ Vector store working - found {len(results)} documents")
        
        return True
    except Exception as e:
        print(f"✗ Vector store test failed: {e}")
        return False

async def test_gemini_service():
    """Test Gemini service"""
    print("\nTesting Gemini service...")
    from services.gemini_service import GeminiService
    from services.document_processor import Document
    
    try:
        gemini_service = GeminiService()
        
        # Create test document
        test_doc = Document(
            "Insurance companies are not liable when vehicles operate without valid permits",
            "test_case.txt",
            "chunk1",
            {}
        )
        
        # Test answer generation
        answer, citations = await gemini_service.generate_answer_with_citations(
            "What is the liability of insurance companies?",
            [test_doc]
        )
        
        print(f"✓ Gemini service working")
        print(f"  - Answer length: {len(answer)}")
        print(f"  - Citations count: {len(citations)}")
        
        return True
    except Exception as e:
        print(f"✗ Gemini service test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("=== Legal RAG Backend Tests ===\n")
    
    tests = [
        test_initialization,
        test_document_processing,
        test_embeddings,
        test_vector_store,
        test_gemini_service
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if await test():
            passed += 1
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! The application is ready.")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")

if __name__ == "__main__":
    asyncio.run(main())