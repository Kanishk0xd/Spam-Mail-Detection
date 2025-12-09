

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn
import logging
from datetime import datetime
import os

from app.model import SpamDetector

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Spam Email Detection API",
    description="API for detecting spam emails using ML",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize spam detector
detector = SpamDetector()

# Request/Response models
class EmailRequest(BaseModel):
    text: str = Field(..., description="Email text content", min_length=1)
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Congratulations! You have won a free prize. Click here now!"
            }
        }

class BatchEmailRequest(BaseModel):
    emails: List[str] = Field(..., description="List of email texts", min_items=1, max_items=100)
    
    class Config:
        json_schema_extra = {
            "example": {
                "emails": [
                    "You have won a lottery!",
                    "Can we schedule a meeting tomorrow?"
                ]
            }
        }

class PredictionResponse(BaseModel):
    text: str
    is_spam: bool
    label: str
    confidence: float
    spam_probability: float
    ham_probability: float
    timestamp: str
    
class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_emails: int
    spam_count: int
    ham_count: int

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str

# Endpoints
@app.get("/", response_class=FileResponse)
async def serve_frontend():
    """Serve the frontend dashboard"""
    frontend_path = "frontend/index.html"
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    return {"message": "Spam Detection API is running. Use /docs for API documentation."}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": detector.is_loaded(),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/predict_spam", response_model=PredictionResponse)
async def predict_spam(request: EmailRequest):
    """
    Predict if a single email is spam
    
    - **text**: The email content to analyze
    
    Returns prediction with confidence scores
    """
    try:
        logger.info(f"Received prediction request for text: {request.text[:50]}...")
        
        result = detector.predict(request.text)
        
        response = {
            "text": request.text,
            "is_spam": result["is_spam"],
            "label": result["label"],
            "confidence": result["confidence"],
            "spam_probability": result["spam_probability"],
            "ham_probability": result["ham_probability"],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info(f"Prediction: {result['label']} (confidence: {result['confidence']:.4f})")
        
        return response
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict_spam_batch", response_model=BatchPredictionResponse)
async def predict_spam_batch(request: BatchEmailRequest):
    """
    Predict spam for multiple emails in batch
    
    - **emails**: List of email texts (max 100)
    
    Returns predictions for all emails with summary statistics
    """
    try:
        logger.info(f"Received batch prediction request for {len(request.emails)} emails")
        
        predictions = []
        spam_count = 0
        
        for email_text in request.emails:
            result = detector.predict(email_text)
            
            predictions.append({
                "text": email_text,
                "is_spam": result["is_spam"],
                "label": result["label"],
                "confidence": result["confidence"],
                "spam_probability": result["spam_probability"],
                "ham_probability": result["ham_probability"],
                "timestamp": datetime.utcnow().isoformat()
            })
            
            if result["is_spam"]:
                spam_count += 1
        
        response = {
            "predictions": predictions,
            "total_emails": len(request.emails),
            "spam_count": spam_count,
            "ham_count": len(request.emails) - spam_count
        }
        
        logger.info(f"Batch prediction complete: {spam_count} spam, {len(request.emails) - spam_count} ham")
        
        return response
        
    except Exception as e:
        logger.error(f"Error during batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get API statistics"""
    return {
        "api_version": "1.0.0",
        "model_type": "TF-IDF + Logistic Regression",
        "max_batch_size": 100,
        "timestamp": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)