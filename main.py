# app/main.py
import os
import logging
import requests
import time
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
TIMEOUT_SECONDS = int(os.getenv("TIMEOUT_SECONDS", 25))  # Increased timeout
MAX_INPUT_LENGTH = int(os.getenv("MAX_INPUT_LENGTH", 256))
MODEL_NAME = os.getenv("MODEL_NAME", "distilgpt2")
HF_API_TOKEN = os.getenv("HUGGING_FACE_TOKEN", None)

# Create FastAPI app
app = FastAPI(
    title="LLM Text Generation API",
    description="API for generating text using Hugging Face models",
    version="1.0.0"
)

# Pydantic models
class GenerationRequest(BaseModel):
    prompt: str = Field(..., description="The text prompt to generate from")
    max_length: int = Field(100, description="Maximum length of generated text")
    temperature: float = Field(0.7, description="Temperature for text generation (0.0-1.0)")
    
    class Config:
        schema_extra = {
            "example": {
                "prompt": "Once upon a time in a distant galaxy",
                "max_length": 100,
                "temperature": 0.7
            }
        }

class GenerationResponse(BaseModel):
    prompt: str = Field(..., description="The original prompt")
    generated_text: str = Field(..., description="The generated text")
    model_used: str = Field(..., description="Name of the model used")

# Direct API call to HuggingFace - faster than loading the model
def generate_with_api(prompt, max_length=100, temperature=0.7):
    if not HF_API_TOKEN:
        return "Error: No HuggingFace API token found. Please set the HUGGING_FACE_TOKEN environment variable."
    
    # Limit max_length to avoid timeouts
    if max_length > 150:
        logger.warning(f"Reducing max_length from {max_length} to 150 to avoid timeouts")
        max_length = 150
    
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_length": max_length,
            "temperature": temperature,
            "return_full_text": True
        }
    }
    
    start_time = time.time()
    logger.info(f"Starting API call to HuggingFace at {start_time}")
    
    try:
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{MODEL_NAME}",
            headers=headers,
            json=payload,
            timeout=TIMEOUT_SECONDS
        )
        
        end_time = time.time()
        logger.info(f"API call completed in {end_time - start_time:.2f} seconds")
        
        if response.status_code == 200:
            result = response.json()
            # Handle different response formats
            if isinstance(result, list) and len(result) > 0:
                if "generated_text" in result[0]:
                    return result[0]["generated_text"]
                return result[0]
            return str(result)
        else:
            logger.error(f"HuggingFace API error: {response.status_code} - {response.text}")
            return f"Error: HuggingFace API returned status code {response.status_code}: {response.text}"
    
    except requests.exceptions.Timeout:
        logger.error(f"HuggingFace API timeout after {time.time() - start_time:.2f} seconds")
        return f"Error: Request timed out after {TIMEOUT_SECONDS} seconds. Try with a smaller max_length value."
    
    except Exception as e:
        logger.error(f"Error in generate_with_api: {str(e)}")
        return f"Error: {str(e)}"

# Simple in-memory cache for faster responses
text_cache = {}

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to the LLM Text Generation API",
        "docs": "/docs",
        "model": MODEL_NAME,
        "max_input_length": MAX_INPUT_LENGTH,
        "timeout": TIMEOUT_SECONDS
    }

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    """Generate text based on the provided prompt."""
    logger.info(f"Received generation request with prompt: {request.prompt[:30]}...")
    
    # Validate input length
    if len(request.prompt) > MAX_INPUT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Prompt too long. Maximum length is {MAX_INPUT_LENGTH} characters"
        )
    
    # Check cache for identical requests to save time
    cache_key = f"{request.prompt}_{request.max_length}_{request.temperature}"
    if cache_key in text_cache:
        logger.info("Returning cached response")
        return text_cache[cache_key]
    
    try:
        # Generate text directly with API call - much faster
        generated_text = generate_with_api(
            prompt=request.prompt,
            max_length=request.max_length,
            temperature=request.temperature
        )
        
        # Check if error message was returned
        if generated_text.startswith("Error:"):
            raise HTTPException(status_code=500, detail=generated_text)
        
        # Create response
        response = GenerationResponse(
            prompt=request.prompt,
            generated_text=generated_text,
            model_used=MODEL_NAME
        )
        
        # Cache the result
        text_cache[cache_key] = response
        
        # Return response
        return response
    
    except Exception as e:
        logger.error(f"Error generating text: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "max_input_length": MAX_INPUT_LENGTH,
        "timeout": TIMEOUT_SECONDS
    }

@app.get("/test-auth")
async def test_auth():
    """Test authentication with HuggingFace."""
    # Test HuggingFace connection
    if not HF_API_TOKEN:
        return {
            "status": "error",
            "message": "No HuggingFace API token found in environment variables"
        }
    
    try:
        # Test auth with HF API directly
        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        response = requests.get(
            f"https://api-inference.huggingface.co/models/{MODEL_NAME}", 
            headers=headers,
            timeout=5
        )
        
        if response.status_code == 200:
            return {
                "status": "success",
                "message": "HuggingFace authentication successful",
                "model_info": response.json()
            }
        else:
            return {
                "status": "error",
                "message": f"HuggingFace authentication failed with status code {response.status_code}",
                "details": response.text
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error testing HuggingFace authentication: {str(e)}"
        }

@app.get("/debug")
async def debug(request: Request):
    """Debug endpoint showing request headers and environment."""
    # Get headers
    headers_dict = dict(request.headers.items())
    
    # Get environment status
    env_status = {
        "HF_TOKEN": "present" if HF_API_TOKEN else "missing",
        "MODEL_NAME": MODEL_NAME,
        "TIMEOUT": TIMEOUT_SECONDS,
        "MAX_LENGTH": MAX_INPUT_LENGTH
    }
    
    return {
        "message": "Debug information",
        "headers": headers_dict,
        "environment": env_status
    }

# Simpler generate endpoint with smaller outputs for testing
@app.post("/generate-quick")
async def generate_text_quick(request: GenerationRequest):
    """Generate text quickly with smaller outputs."""
    # Force smaller output to avoid timeouts
    max_length = min(request.max_length, 50)
    
    try:
        generated_text = generate_with_api(
            prompt=request.prompt,
            max_length=max_length,
            temperature=request.temperature
        )
        
        return {
            "prompt": request.prompt,
            "generated_text": generated_text,
            "model_used": MODEL_NAME,
            "note": "Using reduced max_length to prevent timeouts"
        }
    
    except Exception as e:
        logger.error(f"Error in quick generation: {str(e)}")
        return {"error": str(e)}
