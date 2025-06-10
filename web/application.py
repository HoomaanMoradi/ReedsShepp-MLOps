"""FastAPI application for demand prediction using a trained MLP model.

This application provides an endpoint to predict demand based on input features using a
pre-trained MLP model. It accepts input features within specified ranges and returns the
top-k predicted classes with their probabilities.
"""

import os
from pathlib import Path
import numpy as np
import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, confloat
from typing import List, Tuple

from src.config_reader import read_config

# Load configuration
config_path = "config/config.yaml"
model_training_config = read_config(config_path)["model_training"]
web_config = read_config(config_path)["web"]

# Load the trained model
model_dir = web_config["model_output_dir"]
model_name = web_config["model_name"]
model = joblib.load(Path(model_dir) / model_name)

# Initialize FastAPI application
app = FastAPI()


class DemandRequest(BaseModel):
    """Request model for demand prediction.

    Attributes:
        input1: Feature 1 value (angle in radians, 0 to π)
        input2: Feature 2 value (angle in radians, -π to π)
        input3: Feature 3 value (distance, 0 to 8.5)
    """
    input1: confloat(ge=0, le=3.1416)
    input2: confloat(ge=-3.1416, le=3.1416)
    input3: confloat(ge=0, le=8.5)


class Prediction(BaseModel):
    """Model for individual prediction result.

    Attributes:
        class_index: Index of the predicted class
        probability: Probability of the predicted class
    """
    class_index: int
    probability: float


class DemandResponse(BaseModel):
    """Response model containing top-k predictions.

    Attributes:
        predictions: List of Prediction objects containing class indices and probabilities
    """
    predictions: List[Prediction]


@app.post("/predict", response_model=DemandResponse)
def predict_demand(request: DemandRequest) -> DemandResponse:
    """Predict demand based on input features.

    Args:
        request: DemandRequest object containing input features

    Returns:
        DemandResponse: Response containing top-k predictions with probabilities
    """
    # Prepare input features as DataFrame
    features = pd.DataFrame(
        [
            {
                "input1": request.input1,
                "input2": request.input2,
                "input3": request.input3,
            }
        ]
    )

    # Get probabilities for all classes
    probabilities = model.predict_proba(features)[0]
    
    # Get indices of top_k classes with highest probabilities
    top_k_indices = np.argsort(probabilities)[-model_training_config.get("top_k", 5):][::-1]
    
    # Get the corresponding probabilities
    top_k_probs = probabilities[top_k_indices]
    
    # Create prediction objects
    predictions = [
        Prediction(
            class_index=int(idx),
            probability=float(prob)
        )
        for idx, prob in zip(top_k_indices, top_k_probs)
    ]
    
    return DemandResponse(predictions=predictions)



if __name__ == "__main__":
    uvicorn.run(
        "web.application:app",
        host=os.environ.get("WEB_HOST", web_config["host"]),
        port=int(os.environ.get("WEB_PORT", web_config["port"])),
        reload=True,
    )
