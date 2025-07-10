"""
Main entry point for the Legal RAG Backend
"""
from flask_app import app

# WSGI application for gunicorn
application = app
