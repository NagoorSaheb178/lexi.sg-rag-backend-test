#!/usr/bin/env python3
"""
ASGI configuration for FastAPI application
"""
from app import app

# This is the ASGI application that will be used by the server
application = app