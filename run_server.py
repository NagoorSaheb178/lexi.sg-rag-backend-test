#!/usr/bin/env python3
"""
Simple server runner for FastAPI application
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)