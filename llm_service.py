import os
import logging
import requests
import json
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get API token
HF_API_TOKEN = os.getenv("HUGGING_FACE_TOKEN", None)
if not HF_API_TOKEN:
    logger.warning("⚠️ No HuggingFace API token found. API calls will likely fail with 401 Unauthorized.")
else:
    # Log partial token for debugging (safely)
    token_preview = HF_API_TOKEN[:4] + "..." if HF_API_TOKEN else "None"
    logger.info(f"HuggingFace API token loaded: {token_preview}")

class LLMService:
    def __init__(self, model_name="distilgpt2"):
        """Initialize the LLM service with a specific model."""
        self.model_name = model_name
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        
        # Set up headers with token if available
        if HF_API_TOKEN:
            self.headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
            logger.info("Authorization header configured for HuggingFace API")
        else:
            self.headers = {}
            logger.warning("No authorization header set - API calls may fail")
            
        logger.info(f"LLM Service initialized with model {model_name} using HuggingFace Inference API")
        
    def generate(self, prompt, max_length=100, temperature=0.7):
        """Generate text based on the provided prompt using HuggingFace Inference API."""
        try:
            logger.info(f"Sending request to HuggingFace API with prompt: {prompt[:50]}...")
            logger.info(f"Request headers: {json.dumps({k: '***' if k == 'Authorization' else v for k, v in self.headers.items()})}")
            
            # Prepare payload
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_length": min(max_length, 256),
                    "temperature": max(0.1, min(temperature, 1.0)),
                    "do_sample": True,
                    "return_full_text": False
                }
            }
            
            # Make API request
            response = requests.post(
                self.api_url, 
                headers=self.headers, 
                json=payload
            )
            
            # Log response info
            logger.info(f"Response status code: {response.status_code}")
            
            # Handle response
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    if "generated_text" in result[0]:
                        return result[0]["generated_text"]
                    else:
                        return result[0]
                else:
                    return str(result)
            elif response.status_code == 401:
                error_msg = "Authentication failed: Invalid or missing HuggingFace token"
                logger.error(error_msg)
                logger.error(f"Response body: {response.text}")
                return f"Error: {error_msg}. Please check your API configuration."
            elif response.status_code == 503:
                # Model is loading
                logger.warning("Model is loading on HuggingFace servers. Please retry in a few seconds.")
                return f"Model is currently loading. Please try again in a few seconds. (Prompt: {prompt})"
            else:
                logger.error(f"Error from HuggingFace API: {response.status_code} - {response.text}")
                return f"Error generating text. Please try again later. (Status code: {response.status_code})"
            
        except Exception as e:
            logger.error(f"Error in text generation: {str(e)}")
            return f"Error: {str(e)}"