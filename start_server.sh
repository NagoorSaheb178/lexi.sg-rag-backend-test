#!/bin/bash

# Legal RAG Backend Server Startup Script

echo "Starting Legal RAG Backend..."
echo "Gemini API Key: ${GEMINI_API_KEY:0:20}..."

# Change to the application directory
cd /home/runner/workspace

# Kill any existing processes
pkill -f uvicorn || true

# Start the FastAPI server using uvicorn
uvicorn app:app --host 0.0.0.0 --port 5000 --reload --log-level info