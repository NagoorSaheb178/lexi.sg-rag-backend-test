#!/usr/bin/env python3
"""
Test the API endpoints directly
"""
import asyncio
import json
import requests
from fastapi.testclient import TestClient
from app import app

# Create test client
client = TestClient(app)

def test_health_endpoint():
    """Test health endpoint"""
    print("Testing /health endpoint...")
    try:
        response = client.get("/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health endpoint failed: {e}")
        return False

def test_query_endpoint():
    """Test query endpoint"""
    print("\nTesting /query endpoint...")
    try:
        query_data = {
            "query": "Is an insurance company liable to pay compensation if a transport vehicle involved in an accident was being used without a valid permit?"
        }
        
        response = client.post("/query", json=query_data)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Answer length: {len(result['answer'])}")
            print(f"Citations count: {len(result['citations'])}")
            print(f"Answer preview: {result['answer'][:200]}...")
            
            if result['citations']:
                print(f"First citation: {result['citations'][0]['text'][:100]}...")
            
            return True
        else:
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"Query endpoint failed: {e}")
        return False

def test_document_count():
    """Test document count endpoint"""
    print("\nTesting /documents/count endpoint...")
    try:
        response = client.get("/documents/count")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Document count endpoint failed: {e}")
        return False

def main():
    """Run API tests"""
    print("=== Legal RAG Backend API Tests ===\n")
    
    tests = [
        test_health_endpoint,
        test_document_count,
        test_query_endpoint
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n=== API Test Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All API tests passed!")
    else:
        print("⚠️  Some API tests failed.")

if __name__ == "__main__":
    main()