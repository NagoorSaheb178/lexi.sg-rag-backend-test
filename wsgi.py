"""
WSGI/ASGI compatibility layer for Legal RAG Backend
"""
import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from app import app

# For WSGI compatibility, we need to wrap the FastAPI app
# This is a simple solution that will work with both WSGI and ASGI servers

# Export the app for both WSGI and ASGI
application = app
asgi_app = app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)