#index.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

# Import the app from main.py
from main import app

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Log startup
logger.info("API starting up")

# This is for Vercel serverless deployment