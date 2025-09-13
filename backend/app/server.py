from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import numpy as np
from PIL import Image
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
USE_DUMMY_PREDICTIONS = True

# Load model (only if not using dummy predictions)
if not USE_DUMMY_PREDICTIONS:
    import joblib
    model = joblib.load("app/model.joblib")

# Class names for vehicle condition classification
class_names = np.array(["clean", "dirty", "scratchless", "scratched"])

# Vehicle sides mapping for better logging
VEHICLE_SIDES = ["front", "back", "left", "right"]

# Initialize FastAPI app
app = FastAPI(
    title="inDrive Vehicle Inspector API",
    description="AI-powered vehicle condition assessment for inDrive drivers",
    version="2.0.0"
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def preprocess_image(image: Image.Image, model_type="sklearn"):
    """
    Preprocess image for model prediction.
    
    Args:
        image: PIL Image object
        model_type: Type of model ("sklearn" or "cnn")
    
    Returns:
        Preprocessed image array
    """
    img_array = np.array(image) / 255.0
    
    if model_type == "sklearn":
        return img_array.reshape(1, -1)
    elif model_type == "cnn":
        return np.expand_dims(img_array, axis=0)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

async def process_single_image(file: UploadFile, side_name: str = "unknown") -> dict:
    """
    Process a single vehicle image and return prediction results.
    
    Args:
        file: Uploaded image file
        side_name: Name of the vehicle side (front, back, left, right)
    
    Returns:
        Dictionary with prediction results
    """
    try:
        # Validate file
        if not file or not file.filename:
            return {"error": f"No file uploaded for {side_name} side"}
        
        if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
            return {"error": f"Invalid file type for {side_name} side. Only JPG and PNG are supported."}

        # Read and validate file size
        MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            return {"error": f"File too large for {side_name} side. Max size is 5 MB."}

        logger.info(f"Processing {side_name} side image: {file.filename}, size: {len(contents)} bytes")
        
        # Process image
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((224, 224))
        
        if USE_DUMMY_PREDICTIONS:
            # Generate deterministic dummy predictions based on filename
            np.random.seed(hash(file.filename) % 2**32)
            dummy_probs = np.random.dirichlet([2, 1, 1, 1])
            prediction_idx = np.argmax(dummy_probs)
            
            class_name = class_names[prediction_idx]
            confidence = float(dummy_probs[prediction_idx])
            trust_score = round(confidence * 100, 2)
            
            logger.info(f"DUMMY Prediction for {side_name}: {class_name}, Trust Score: {trust_score}%")
            
            return {
                "side": side_name,
                "filename": file.filename,
                "predicted_class": class_name,
                "confidence": confidence,
                "trust_score": trust_score,
                "all_scores": dict(zip(class_names, dummy_probs.tolist())),
                "explanation": f"The {side_name} side was classified as '{class_name}' with {trust_score}% confidence.",
                "note": "Using dummy predictions - replace with real model"
            }
        
        else:
            # Real model predictions
            img_array = preprocess_image(image, model_type="sklearn")
            
            prediction = model.predict(img_array)
            prediction_proba = model.predict_proba(img_array)
            
            class_name = class_names[prediction[0]]
            confidence = float(np.max(prediction_proba))
            trust_score = round(confidence * 100, 2)
            
            logger.info(f"Real Prediction for {side_name}: {class_name}, Trust Score: {trust_score}%")
            
            return {
                "side": side_name,
                "filename": file.filename,
                "predicted_class": class_name,
                "confidence": confidence,
                "trust_score": trust_score,
                "all_scores": dict(zip(class_names, prediction_proba[0].tolist())),
                "explanation": f"The {side_name} side was classified as '{class_name}' with {trust_score}% confidence."
            }
        
    except Exception as e:
        logger.error(f"Error processing {side_name} side image {file.filename if file else 'unknown'}: {str(e)}")
        return {
            "side": side_name,
            "filename": file.filename if file else "unknown",
            "error": f"Failed to process {side_name} side image",
            "details": str(e),
            "hint": "Ensure the uploaded file is a valid image (jpg/png)."
        }

@app.get("/")
def read_root():
    """API information endpoint."""
    return {
        "message": "inDrive Vehicle Inspector API",
        "description": "AI-powered 4-side vehicle condition assessment",
        "version": "2.0.0",
        "endpoints": {
            "/inspect-vehicle": "Complete 4-side vehicle inspection",
            "/health": "System health check",
            "/docs": "API documentation"
        },
        "docs_url": "/docs"
    }

@app.get("/health")
def health_check():
    """System health check endpoint."""
    return {
        "status": "healthy",
        "model_status": "dummy" if USE_DUMMY_PREDICTIONS else "loaded",
        "class_labels": class_names.tolist(),
        "required_photos": 4,
        "vehicle_sides": VEHICLE_SIDES,
        "version": "2.0.0",
        "max_file_size": "5MB",
        "supported_formats": ["JPG", "JPEG", "PNG"]
    }

@app.post("/inspect-vehicle")
async def inspect_vehicle(files: List[UploadFile] = File(...)):
    """
    Complete 4-side vehicle condition inspection.
    
    Analyzes front, back, left, and right sides of the vehicle.
    Returns comprehensive assessment with individual and overall trust scores.
    
    Args:
        files: List of exactly 4 image files (front, back, left, right sides)
    
    Returns:
        Complete vehicle inspection results
    """
    try:
        # Validate number of files
        if len(files) != 4:
            logger.warning(f"Invalid number of files received: {len(files)} (expected 4)")
            return {
                "error": "Exactly 4 photos are required",
                "details": "Please upload photos of all vehicle sides: front, back, left, and right",
                "received_files": len(files),
                "required_files": 4
            }

        logger.info(f"Starting complete vehicle inspection with {len(files)} images")
        
        results = []
        processing_errors = []
        
        # Process each image
        for i, file in enumerate(files):
            side_name = VEHICLE_SIDES[i] if i < len(VEHICLE_SIDES) else f"side_{i+1}"
            result = await process_single_image(file, side_name)
            
            if "error" in result:
                processing_errors.append(result)
            
            results.append(result)
        
        # Calculate overall metrics
        successful_results = [r for r in results if "error" not in r]
        failed_results = [r for r in results if "error" in r]
        
        overall_trust_score = 0
        vehicle_condition_summary = {}
        
        if successful_results:
            # Calculate average trust score
            trust_scores = [r["trust_score"] for r in successful_results]
            overall_trust_score = round(sum(trust_scores) / len(trust_scores), 2)
            
            # Analyze overall vehicle condition
            conditions = [r["predicted_class"] for r in successful_results]
            condition_counts = {}
            for condition in conditions:
                condition_counts[condition] = condition_counts.get(condition, 0) + 1
            
            # Find dominant condition safely
            dominant_condition = None
            if condition_counts:
                dominant_condition = max(condition_counts.keys(), key=lambda x: condition_counts[x])
            
            vehicle_condition_summary = {
                "dominant_condition": dominant_condition,
                "condition_distribution": condition_counts,
                "sides_analyzed": len(successful_results),
                "sides_failed": len(failed_results)
            }
        
        # Generate overall assessment
        inspection_status = "complete" if len(successful_results) == 4 else "partial"
        
        if overall_trust_score >= 90:
            assessment_message = "Excellent! Your vehicle is in outstanding condition."
        elif overall_trust_score >= 80:
            assessment_message = "Great! Your vehicle meets high quality standards."
        elif overall_trust_score >= 70:
            assessment_message = "Good condition with minor areas for improvement."
        elif overall_trust_score >= 60:
            assessment_message = "Acceptable condition, consider some maintenance."
        else:
            assessment_message = "Vehicle needs attention before providing rides."
        
        response_data = {
            "inspection_id": hash(str([f.filename for f in files])) % 10000000,
            "inspection_status": inspection_status,
            "results": results,
            "summary": {
                "total_images_processed": len(files),
                "successful_predictions": len(successful_results),
                "failed_predictions": len(failed_results),
                "overall_trust_score": overall_trust_score,
                "assessment_message": assessment_message,
                "vehicle_condition": vehicle_condition_summary
            },
            "recommendations": [],
            "timestamp": __import__('datetime').datetime.now().isoformat()
        }
        
        # Add recommendations based on results
        if overall_trust_score < 80:
            response_data["recommendations"].append("Consider cleaning your vehicle before providing rides")
        if any(r.get("predicted_class") == "scratched" for r in successful_results):
            response_data["recommendations"].append("Address visible scratches to improve passenger confidence")
        if len(successful_results) < 4:
            response_data["recommendations"].append("Retake failed photos for complete assessment")
        if overall_trust_score >= 90:
            response_data["recommendations"].append("Excellent condition! You're ready for premium rides")
        
        logger.info(f"Vehicle inspection completed. Overall trust score: {overall_trust_score}%")
        return response_data
        
    except Exception as e:
        logger.error(f"Critical error during vehicle inspection: {str(e)}")
        return {
            "error": "Vehicle inspection failed",
            "details": "A critical error occurred during processing",
            "technical_details": str(e),
            "hint": "Please try again with valid image files"
        }

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unexpected errors."""
    logger.error(f"Unexpected error: {str(exc)}")
    return {
        "error": "Internal server error",
        "message": "An unexpected error occurred. Please try again.",
        "status_code": 500
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )