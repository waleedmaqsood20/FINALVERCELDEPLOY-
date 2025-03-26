# app/__init__.py
# This file is intentionally left empty to make the directory a Python package

# app/models.py
from pydantic import BaseModel, Field
from typing import Dict, Any

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
    
    def dict(self, **kwargs) -> Dict[str, Any]:
        """Override dict method for serialization compatibility."""
        return {
            "prompt": self.prompt,
            "max_length": self.max_length,
            "temperature": self.temperature
        }

class GenerationResponse(BaseModel):
    prompt: str = Field(..., description="The original prompt")
    generated_text: str = Field(..., description="The generated text")
    model_used: str = Field(..., description="Name of the model used")
    
    def dict(self, **kwargs) -> Dict[str, Any]:
        """Override dict method for serialization compatibility."""
        return {
            "prompt": self.prompt,
            "generated_text": self.generated_text,
            "model_used": self.model_used
        }

