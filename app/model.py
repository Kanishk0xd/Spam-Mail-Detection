"""
Spam Detector Model Wrapper
Loads the trained pipeline and provides prediction interface
"""

import joblib
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class SpamDetector:
    def __init__(self, model_path: str = "models/spam_pipeline.joblib"):
        """
        Initialize spam detector with trained model
        
        Args:
            model_path: Path to the saved joblib model
        """
        self.model_path = model_path
        self.pipeline = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model pipeline"""
        try:
            model_file = Path(self.model_path)
            
            if not model_file.exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            self.pipeline = joblib.load(self.model_path)
            logger.info(f"Model loaded successfully from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.pipeline is not None
    
    def predict(self, text: str) -> dict:
        """
        Predict if email text is spam
        
        Args:
            text: Email text content
            
        Returns:
            Dictionary with prediction results
        """
        if not self.is_loaded():
            raise RuntimeError("Model is not loaded")
        
        if not text or not text.strip():
            raise ValueError("Email text cannot be empty")
        
        try:
            # Get prediction
            prediction = self.pipeline.predict([text])[0]
            
            # Get probability scores
            probabilities = self.pipeline.predict_proba([text])[0]
            
            # Extract scores
            ham_prob = float(probabilities[0])
            spam_prob = float(probabilities[1])
            
            # Determine label and confidence
            is_spam = bool(prediction == 1)
            label = "SPAM" if is_spam else "HAM"
            confidence = spam_prob if is_spam else ham_prob
            
            result = {
                "is_spam": is_spam,
                "label": label,
                "confidence": confidence,
                "spam_probability": spam_prob,
                "ham_probability": ham_prob
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise
    
    def predict_batch(self, texts: list) -> list:
        """
        Predict spam for multiple emails
        
        Args:
            texts: List of email text contents
            
        Returns:
            List of prediction dictionaries
        """
        if not self.is_loaded():
            raise RuntimeError("Model is not loaded")
        
        results = []
        for text in texts:
            try:
                result = self.predict(text)
                results.append(result)
            except Exception as e:
                logger.error(f"Error predicting text: {str(e)}")
                results.append({
                    "error": str(e),
                    "text": text[:50] + "..."
                })
        
        return results
    
    def get_top_features(self, text: str, top_n: int = 10) -> list:
        """
        Get top contributing features for prediction (optional enhancement)
        
        Args:
            text: Email text content
            top_n: Number of top features to return
            
        Returns:
            List of (feature, weight) tuples
        """
        if not self.is_loaded():
            raise RuntimeError("Model is not loaded")
        
        try:
            # Transform text to TF-IDF features
            tfidf_vector = self.pipeline.named_steps['tfidf'].transform([text])
            
            # Get feature names
            feature_names = self.pipeline.named_steps['tfidf'].get_feature_names_out()
            
            # Get coefficients from logistic regression
            coefficients = self.pipeline.named_steps['classifier'].coef_[0]
            
            # Get non-zero features from the text
            non_zero_indices = tfidf_vector.nonzero()[1]
            
            # Calculate feature contributions
            contributions = []
            for idx in non_zero_indices:
                feature_name = feature_names[idx]
                weight = tfidf_vector[0, idx] * coefficients[idx]
                contributions.append((feature_name, float(weight)))
            
            # Sort by absolute weight
            contributions.sort(key=lambda x: abs(x[1]), reverse=True)
            
            return contributions[:top_n]
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return []