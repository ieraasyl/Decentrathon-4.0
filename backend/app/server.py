import os
import asyncio
import json
import hashlib
import logging
from datetime import datetime
from typing import List, Optional
from PIL import Image
import io
import numpy as np

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import httpx

# Local ML model imports
import torch
import torch.nn as nn
from torchvision import transforms
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Validation configuration
IMAGE_CONFIG = {
    "max_size": 10 * 1024 * 1024,  # 10MB
    "min_size": 10 * 1024,         # 10KB
    "allowed_types": ["image/jpeg", "image/jpg", "image/png", "image/webp"],
    "max_dimension": 4096,
    "min_dimension": 224,
    "required_files": 4,
    "max_files": 4
}

# Vehicle sides mapping (order matters!)
VEHICLE_SIDES = ["front", "back", "left", "right"]

# Modal ML service endpoint from environment variable
MODAL_ML_URL = "https://ieraasyl--indrive-vehicle-inspector-fastapi-app.modal.run/predict"

# Local model paths
CAR_PARTS_MODEL_PATH = "best_car_parts.pt"
CLEAN_DIRTY_MODEL_PATH = "efficientnet_binary_clean_dirty.pth"

# Global variables for local models
local_car_parts_model = None
local_clean_dirty_model = None

# Image preprocessing transforms (same as Modal)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Initialize FastAPI app
app = FastAPI(
    title="inDrive Vehicle Inspector API",
    description="AI-powered vehicle condition assessment with comprehensive validation",
    version="3.0.0"
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ValidationError(Exception):
    def __init__(self, message: str, details: Optional[str] = None):
        self.message = message
        self.details = details
        super().__init__(message)

def validate_image_file(file: UploadFile) -> dict:
    """
    Comprehensive image validation
    """
    errors = []
    warnings = []
    
    # Check content type
    if file.content_type and file.content_type not in IMAGE_CONFIG["allowed_types"]:
        errors.append(f"Invalid file type '{file.content_type}'. Allowed types: {', '.join(IMAGE_CONFIG['allowed_types'])}")
    
    # Try to open and validate image
    try:
        file.file.seek(0)  # Reset file pointer
        content = file.file.read()
        file.file.seek(0)  # Reset again for later use
        
        # Check actual file size
        actual_size = len(content)
        if actual_size > IMAGE_CONFIG["max_size"]:
            errors.append(f"File too large ({actual_size / 1024 / 1024:.2f}MB). Maximum: {IMAGE_CONFIG['max_size'] / 1024 / 1024}MB")
        
        if actual_size < IMAGE_CONFIG["min_size"]:
            errors.append(f"File too small ({actual_size / 1024:.2f}KB). Minimum: {IMAGE_CONFIG['min_size'] / 1024}KB")
        
        # Validate image integrity
        img = Image.open(io.BytesIO(content))
        img.verify()  # Verify image integrity
        
        # Re-open for dimension checks (verify() closes the image)
        img = Image.open(io.BytesIO(content))
        width, height = img.size
        
        # Check dimensions
        if width > IMAGE_CONFIG["max_dimension"] or height > IMAGE_CONFIG["max_dimension"]:
            errors.append(f"Image too large ({width}x{height}). Maximum: {IMAGE_CONFIG['max_dimension']}x{IMAGE_CONFIG['max_dimension']}")
        
        if width < IMAGE_CONFIG["min_dimension"] or height < IMAGE_CONFIG["min_dimension"]:
            errors.append(f"Image too small ({width}x{height}). Minimum: {IMAGE_CONFIG['min_dimension']}x{IMAGE_CONFIG['min_dimension']}")
        
        # Check aspect ratio
        aspect_ratio = width / height
        if aspect_ratio > 3 or aspect_ratio < 0.33:
            warnings.append(f"Unusual aspect ratio ({aspect_ratio:.2f}:1). Vehicle photos work best with standard proportions")
        
        # Check if image is too dark or too bright
        try:
            img_array = np.array(img.convert('L'))  # Convert to grayscale
            mean_brightness = np.mean(img_array)
            
            if mean_brightness < 50:
                warnings.append("Image appears very dark. Consider taking photo in better lighting")
            elif mean_brightness > 200:
                warnings.append("Image appears overexposed. Consider adjusting camera settings")
        except:
            pass  # Skip brightness check if it fails
        
    except Exception as e:
        errors.append(f"Invalid or corrupted image file: {str(e)}")
    
    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }

def detect_duplicate_images(files: List[UploadFile]) -> List[dict]:
    """
    Detect duplicate images using simple hash comparison
    """
    duplicates = []
    file_hashes = {}
    
    for i, file in enumerate(files):
        try:
            file.file.seek(0)
            content = file.file.read()
            file.file.seek(0)
            
            # Create hash of image content
            file_hash = hashlib.md5(content).hexdigest()[:16]  # Use first 16 chars
            
            if file_hash in file_hashes:
                duplicates.append({
                    "file_index": i,
                    "filename": file.filename,
                    "duplicate_of_index": file_hashes[file_hash],
                    "duplicate_of_filename": files[file_hashes[file_hash]].filename
                })
            else:
                file_hashes[file_hash] = i
                
        except Exception as e:
            logger.error(f"Error checking for duplicates in file {i}: {e}")
    
    return duplicates

def categorize_cleanliness(dirty_prob: float) -> str:
    """Convert dirty probability to user-friendly category"""
    if dirty_prob < 0.1:
        return "very clean"
    elif dirty_prob < 0.25:
        return "clean"
    elif dirty_prob < 0.5:
        return "slightly dirty"
    elif dirty_prob < 0.75:
        return "dirty"
    else:
        return "very dirty"

def calculate_cleanliness_score(dirty_prob: float) -> int:
    """Convert dirty probability to 0-100 score (higher = cleaner)"""
    return int((1 - dirty_prob) * 100)

def load_local_models():
    """Load both models locally: YOLO for car parts and EfficientNet for clean/dirty"""
    global local_car_parts_model, local_clean_dirty_model
    
    try:
        # Load YOLO car parts model
        if not os.path.exists(CAR_PARTS_MODEL_PATH):
            raise FileNotFoundError(f"Car parts model not found at {CAR_PARTS_MODEL_PATH}")
        
        local_car_parts_model = YOLO(CAR_PARTS_MODEL_PATH)
        logger.info(f"✅ Car parts model loaded from {CAR_PARTS_MODEL_PATH}")
        
        # Load clean/dirty model
        if not os.path.exists(CLEAN_DIRTY_MODEL_PATH):
            raise FileNotFoundError(f"Clean/dirty model not found at {CLEAN_DIRTY_MODEL_PATH}")
        
        # Handle both full model and state dict loading
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            clean_dirty_model = torch.load(CLEAN_DIRTY_MODEL_PATH, map_location=device)
            if hasattr(clean_dirty_model, 'eval'):
                clean_dirty_model.eval()
                local_clean_dirty_model = clean_dirty_model
            else:
                # It's a state dict, recreate the model
                from torchvision.models import efficientnet_b0
                model = efficientnet_b0(weights=None)
                
                # Handle EfficientNet classifier structure (2 classes: clean, dirty)
                model.classifier = torch.nn.Sequential(
                    torch.nn.Dropout(p=0.2, inplace=True),
                    torch.nn.Linear(1280, 2)  # 2 classes based on saved model
                )
                
                model.load_state_dict(clean_dirty_model)
                local_clean_dirty_model = model.to(device)
                local_clean_dirty_model.eval()
                
        except Exception as model_error:
            logger.error(f"Model loading error: {model_error}")
            raise
        
        logger.info(f"✅ Clean/dirty model loaded from {CLEAN_DIRTY_MODEL_PATH}")
        logger.info(f"✅ Using device: {device}")
        
    except Exception as e:
        logger.error(f"❌ Error loading local models: {e}")
        raise

def predict_car_parts_local(img: Image.Image):
    """Predict car parts using local YOLO model"""
    try:
        if local_car_parts_model is None:
            load_local_models()
        
        results = local_car_parts_model(img)
        detections = []
        
        for r in results:
            for b in r.boxes:
                cls = int(b.cls)
                conf = float(b.conf)
                xyxy = b.xyxy[0].tolist()  # [x1, y1, x2, y2]
                
                # Get the actual class name from the model
                vehicle_side = local_car_parts_model.names.get(cls, f"unknown_part_{cls}")
                
                detections.append({
                    "vehicle_side": vehicle_side,
                    "confidence": conf,
                    "bbox": xyxy,
                    "original_class": cls
                })
        
        return {
            "detections": detections,
            "total_detections": len(detections)
        }
    except Exception as e:
        return {"error": f"Car parts prediction failed: {str(e)}"}

def predict_clean_dirty_local(img: Image.Image):
    """Predict clean/dirty using local EfficientNet model"""
    try:
        if local_clean_dirty_model is None:
            load_local_models()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        img_rgb = img.convert("RGB")
        x = transform(img_rgb).unsqueeze(0).to(device)
        
        with torch.no_grad():
            out = local_clean_dirty_model(x)
            probs = torch.softmax(out, dim=1)
            dirty_prob = probs[0][1].item()  # Get probability of dirty class (index 1)
        
        return {"dirty_prob": dirty_prob}
    except Exception as e:
        return {"error": f"Clean/dirty prediction failed: {str(e)}"}

def validate_vehicle_side_match(expected_side: str, detected_side: str, confidence: float, min_confidence: float = 0.3) -> dict:
    """Validate if detected vehicle side matches expected side"""
    if detected_side == "unknown" or confidence < min_confidence:
        return {
            "is_valid": False,
            "error_type": "detection_failed",
            "message": f"Could not detect vehicle part in {expected_side} photo",
            "suggestion": f"Please retake the {expected_side} photo with better lighting and angle"
        }
    
    # Handle alternative names (e.g., back/rear)
    side_aliases = {
        "back": "rear",
        "rear": "back"
    }
    
    normalized_expected = side_aliases.get(expected_side.lower(), expected_side.lower())
    normalized_detected = side_aliases.get(detected_side.lower(), detected_side.lower())
    
    if normalized_detected == normalized_expected:
        return {
            "is_valid": True,
            "confidence": confidence
        }
    else:
        return {
            "is_valid": False,
            "error_type": "wrong_side",
            "message": f"Expected {expected_side} photo but detected {detected_side}",
            "suggestion": f"Please upload the correct {expected_side} view of your vehicle",
            "detected_side": detected_side,
            "confidence": confidence
        }

async def process_single_image(file: UploadFile, side_name: str = "unknown", max_retries: int = 3) -> dict:
    """
    Send an image to the Modal ML service with validation and retry logic
    """
    for attempt in range(max_retries):
        try:
            file.file.seek(0)  # Reset file pointer
            contents = file.file.read()
            file.file.seek(0)  # Reset for potential retry
            
            # Validate content before sending
            if len(contents) == 0:
                return {
                    "side": side_name,
                    "filename": file.filename,
                    "error": "Empty file",
                    "details": "File contains no data"
                }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                files_data = {"file": (file.filename, contents, file.content_type or "image/jpeg")}
                response = await client.post(MODAL_ML_URL, files=files_data)
            
            if response.status_code != 200:
                logger.error(f"Modal ML service error (attempt {attempt + 1}): {response.text}")
                if attempt == max_retries - 1:
                    return {
                        "side": side_name,
                        "filename": file.filename,
                        "error": "ML service failed",
                        "details": f"HTTP {response.status_code}: {response.text}",
                        "attempts": attempt + 1
                    }
                await asyncio.sleep(1)
                continue
            
            prediction = response.json()
            
            # Validate Modal ML service response format
            required_fields = ["car_parts_model", "clean_dirty_model", "status"]
            missing_fields = [field for field in required_fields if field not in prediction]
            
            if missing_fields:
                return {
                    "side": side_name,
                    "filename": file.filename,
                    "error": "Invalid ML response",
                    "details": f"Missing fields: {missing_fields}"
                }
            
            if prediction.get("status") != "success":
                return {
                    "side": side_name,
                    "filename": file.filename,
                    "error": "ML processing failed",
                    "details": str(prediction)
                }
            
            # Extract car parts detection
            car_parts = prediction["car_parts_model"]
            detected_parts = car_parts.get("detections", [])
            
            # Extract clean/dirty classification
            clean_dirty = prediction["clean_dirty_model"]
            dirty_prob = clean_dirty.get("dirty_prob", 0.5)
            
            # Create user-friendly response
            cleanliness_category = categorize_cleanliness(dirty_prob)
            cleanliness_score = calculate_cleanliness_score(dirty_prob)
            
            # Determine dominant detected vehicle side
            detected_side = "unknown"
            max_confidence = 0
            best_detection = None
            
            for detection in detected_parts:
                if detection["confidence"] > max_confidence:
                    max_confidence = detection["confidence"]
                    detected_side = detection["vehicle_side"]
                    best_detection = detection
            
            # Validate that detected side matches expected side
            side_validation = validate_vehicle_side_match(side_name, detected_side, max_confidence)
            
            logger.info(f"Prediction for {side_name}: cleanliness={cleanliness_category} ({dirty_prob:.3f}), detected_side={detected_side}, valid={side_validation['is_valid']}")
            
            result = {
                "side": side_name,
                "filename": file.filename,
                "processing_attempts": attempt + 1,
                "detected_vehicle_side": detected_side,
                "detection_confidence": max_confidence,
                "side_validation": side_validation,
                "cleanliness_category": cleanliness_category,
                "cleanliness_score": cleanliness_score,
                "dirty_probability": round(dirty_prob, 3),
                "car_parts_detections": detected_parts,
                "total_detections": len(detected_parts),
                "best_detection": best_detection,
                "raw_prediction": prediction  # Include full response for debugging
            }
            
            # Add validation error if side doesn't match
            if not side_validation["is_valid"]:
                result["validation_error"] = {
                    "error_type": side_validation["error_type"],
                    "message": side_validation["message"],
                    "suggestion": side_validation["suggestion"]
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error calling ML service (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                return {
                    "side": side_name,
                    "filename": file.filename if file else "unknown",
                    "error": "Failed to call ML service",
                    "details": str(e),
                    "attempts": attempt + 1
                }
            await asyncio.sleep(1)
    
    # Fallback return (should never reach here due to the logic above, but ensures all code paths return)
    return {
        "side": side_name,
        "filename": file.filename if file else "unknown",
        "error": "Max retries exceeded",
        "details": f"Failed after {max_retries} attempts",
        "attempts": max_retries
    }

# ------------------------
# Routes
# ------------------------

@app.get("/")
def read_root():
    """API info endpoint."""
    return {
        "message": "inDrive Vehicle Inspector API",
        "description": "Backend orchestrator with comprehensive validation (ML handled by Modal)",
        "version": "3.0.0",
        "endpoints": {
            "/inspect-vehicle": "Complete 4-side vehicle inspection with validation",
            "/health": "System health check",
            "/docs": "API documentation"
        },
        "validation_features": [
            "Image integrity validation",
            "Dimension and aspect ratio checks",
            "Duplicate image detection",
            "Brightness analysis",
            "Comprehensive error reporting"
        ]
    }

@app.get("/health")
async def health_check():
    """Enhanced health check with validation configuration"""
    ml_service_status = "unknown"
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            ml_health_url = MODAL_ML_URL.replace('/predict', '/health')
            response = await client.get(ml_health_url)
            if response.status_code == 200:
                ml_service_status = "healthy"
            else:
                ml_service_status = "unhealthy"
    except Exception as e:
        ml_service_status = f"unreachable: {str(e)}"
    
    # Check local models
    local_models_status = {
        "car_parts_available": os.path.exists(CAR_PARTS_MODEL_PATH),
        "clean_dirty_available": os.path.exists(CLEAN_DIRTY_MODEL_PATH),
        "car_parts_loaded": local_car_parts_model is not None,
        "clean_dirty_loaded": local_clean_dirty_model is not None,
        "torch_cuda_available": torch.cuda.is_available()
    }
    
    return {
        "status": "healthy",
        "ml_service_url": MODAL_ML_URL,
        "ml_service_status": ml_service_status,
        "local_models": local_models_status,
        "validation_config": {
            "required_photos": IMAGE_CONFIG["required_files"],
            "max_file_size_mb": IMAGE_CONFIG["max_size"] / 1024 / 1024,
            "min_file_size_kb": IMAGE_CONFIG["min_size"] / 1024,
            "allowed_types": IMAGE_CONFIG["allowed_types"],
            "max_dimension": IMAGE_CONFIG["max_dimension"],
            "min_dimension": IMAGE_CONFIG["min_dimension"]
        },
        "vehicle_sides": VEHICLE_SIDES,
        "version": "3.0.0",
        "supported_formats": ["JPG", "JPEG", "PNG", "WebP"],
        "endpoints": {
            "/predict-local": "Single image prediction using local models",
            "/inspect-vehicle": "Full 4-side inspection using Modal API",
            "/inspect-vehicle-local": "Full 4-side inspection using local models"
        }
    }

@app.post("/predict-local")
async def predict_local(file: UploadFile = File(...)):
    """
    Local ML inference endpoint using models directly (similar to Modal)
    """
    try:
        # Load models on first request
        if local_car_parts_model is None or local_clean_dirty_model is None:
            load_local_models()
        
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Get predictions from both models
        car_parts_result = predict_car_parts_local(img)
        clean_dirty_result = predict_clean_dirty_local(img)
        
        # Check for errors
        if "error" in car_parts_result:
            return {
                "error": "Car parts prediction failed",
                "details": car_parts_result["error"],
                "filename": file.filename
            }
        
        if "error" in clean_dirty_result:
            return {
                "error": "Clean/dirty prediction failed", 
                "details": clean_dirty_result["error"],
                "filename": file.filename
            }
        
        return {
            "filename": file.filename,
            "car_parts_model": car_parts_result,
            "clean_dirty_model": clean_dirty_result,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Local prediction error: {e}")
        return {
            "error": "Failed to process image locally",
            "details": str(e),
            "filename": file.filename or "unknown"
        }

async def process_single_image_local(file: UploadFile, side_name: str = "unknown") -> dict:
    """
    Process image using local models with validation
    """
    try:
        contents = await file.read()
        file.file.seek(0)  # Reset for potential retry
        
        if len(contents) == 0:
            return {
                "side": side_name,
                "filename": file.filename,
                "error": "Empty file",
                "details": "File contains no data"
            }
        
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Get predictions from both models
        car_parts_result = predict_car_parts_local(img)
        clean_dirty_result = predict_clean_dirty_local(img)
        
        # Check for errors
        if "error" in car_parts_result:
            return {
                "side": side_name,
                "filename": file.filename,
                "error": "Car parts prediction failed",
                "details": car_parts_result["error"]
            }
        
        if "error" in clean_dirty_result:
            return {
                "side": side_name,
                "filename": file.filename,
                "error": "Clean/dirty prediction failed", 
                "details": clean_dirty_result["error"]
            }
        
        # Extract results
        car_parts = car_parts_result
        detected_parts = car_parts.get("detections", [])
        
        clean_dirty = clean_dirty_result
        dirty_prob = clean_dirty.get("dirty_prob", 0.5)
        
        # Create user-friendly response
        cleanliness_category = categorize_cleanliness(dirty_prob)
        cleanliness_score = calculate_cleanliness_score(dirty_prob)
        
        # Determine dominant detected vehicle side
        detected_side = "unknown"
        max_confidence = 0
        best_detection = None
        
        for detection in detected_parts:
            if detection["confidence"] > max_confidence:
                max_confidence = detection["confidence"]
                detected_side = detection["vehicle_side"]
                best_detection = detection
        
        # Validate that detected side matches expected side
        side_validation = validate_vehicle_side_match(side_name, detected_side, max_confidence)
        
        logger.info(f"Local prediction for {side_name}: cleanliness={cleanliness_category} ({dirty_prob:.3f}), detected_side={detected_side}, valid={side_validation['is_valid']}")
        
        result = {
            "side": side_name,
            "filename": file.filename,
            "processing_attempts": 1,
            "detected_vehicle_side": detected_side,
            "detection_confidence": max_confidence,
            "side_validation": side_validation,
            "cleanliness_category": cleanliness_category,
            "cleanliness_score": cleanliness_score,
            "dirty_probability": round(dirty_prob, 3),
            "car_parts_detections": detected_parts,
            "total_detections": len(detected_parts),
            "best_detection": best_detection,
            "inference_method": "local"
        }
        
        # Add validation error if side doesn't match
        if not side_validation["is_valid"]:
            result["validation_error"] = {
                "error_type": side_validation["error_type"],
                "message": side_validation["message"],
                "suggestion": side_validation["suggestion"]
            }
        
        return result
        
    except Exception as e:
        return {
            "side": side_name,
            "filename": file.filename if file else "unknown",
            "error": "Failed to process image locally",
            "details": str(e),
            "inference_method": "local"
        }

@app.post("/inspect-vehicle-local")
async def inspect_vehicle_local(
    files: List[UploadFile] = File(...),
    metadata: Optional[str] = Form(None)
):
    """
    Enhanced vehicle inspection using local models with comprehensive validation
    """
    try:
        # Load models on first request
        if local_car_parts_model is None or local_clean_dirty_model is None:
            load_local_models()
        
        # Parse metadata if provided
        request_metadata = {}
        if metadata:
            try:
                request_metadata = json.loads(metadata)
            except json.JSONDecodeError:
                logger.warning("Invalid metadata JSON provided")
        
        # Basic count validation (same as original)
        if len(files) == 0:
            raise ValidationError(
                "No files uploaded",
                "Please upload at least one vehicle photo to continue"
            )
        
        if len(files) > IMAGE_CONFIG["max_files"]:
            raise ValidationError(
                f"Too many files uploaded ({len(files)})",
                f"Maximum {IMAGE_CONFIG['max_files']} files allowed"
            )
        
        if len(files) < IMAGE_CONFIG["required_files"]:
            missing_count = IMAGE_CONFIG["required_files"] - len(files)
            missing_sides = VEHICLE_SIDES[len(files):]
            raise ValidationError(
                f"Incomplete upload ({len(files)}/{IMAGE_CONFIG['required_files']} files)",
                f"Please upload photos for: {', '.join(missing_sides)}"
            )
        
        logger.info(f"Starting local validation for {len(files)} files")
        
        # Validate each file (same as original)
        validation_results = []
        all_warnings = []
        
        for i, file in enumerate(files):
            side_name = VEHICLE_SIDES[i] if i < len(VEHICLE_SIDES) else f"side_{i+1}"
            
            validation = validate_image_file(file)
            validation_results.append({
                "file_index": i,
                "filename": file.filename,
                "side": side_name,
                **validation
            })
            
            if not validation["is_valid"]:
                error_details = "; ".join(validation["errors"])
                raise ValidationError(
                    f"Invalid image for {side_name}",
                    f"{file.filename}: {error_details}"
                )
            
            all_warnings.extend(validation["warnings"])
        
        # Check for duplicate images (same as original)
        duplicates = detect_duplicate_images(files)
        if duplicates:
            duplicate_info = ", ".join([
                f"{dup['filename']} is duplicate of {dup['duplicate_of_filename']}"
                for dup in duplicates
            ])
            raise ValidationError(
                "Duplicate images detected",
                f"Please use different photos for each vehicle side: {duplicate_info}"
            )
        
        logger.info("All files passed validation, proceeding with local ML analysis")
        
        # Process images through local models
        results = []
        for i, file in enumerate(files):
            side_name = VEHICLE_SIDES[i]
            result = await process_single_image_local(file, side_name)
            results.append(result)
        
        # The rest of the processing is the same as the original inspect_vehicle function
        # Analyze results
        successful = [r for r in results if "error" not in r]
        failed = [r for r in results if "error" in r]
        
        # Separate validation errors from processing errors
        validation_failed = [r for r in successful if r.get("validation_error")]
        fully_successful = [r for r in successful if not r.get("validation_error")]
        
        # If any ML processing failed, include validation warnings
        if failed:
            logger.warning(f"{len(failed)} files failed local ML processing")
        if validation_failed:
            logger.warning(f"{len(validation_failed)} files failed side validation")
        
        # Create structured response for each vehicle side
        side_results = {}
        for result in results:
            side = result["side"]
            side_results[side] = {
                "filename": result["filename"],
                "processing_status": "success" if "error" not in result else "failed",
                "validation_status": "passed" if not result.get("validation_error") else "failed",
                "cleanliness_category": result.get("cleanliness_category"),
                "cleanliness_score": result.get("cleanliness_score"),
                "detected_vehicle_side": result.get("detected_vehicle_side"),
                "detection_confidence": result.get("detection_confidence", 0),
                "total_detections": result.get("total_detections", 0),
                "validation_error": result.get("validation_error"),
                "processing_error": result.get("error"),
                "inference_method": "local"
            }
        
        # Calculate overall metrics (only from fully successful results)
        overall_cleanliness_score = 0
        vehicle_condition_summary = {}
        
        if fully_successful:
            cleanliness_scores = [r["cleanliness_score"] for r in fully_successful if "cleanliness_score" in r]
            if cleanliness_scores:
                overall_cleanliness_score = round(sum(cleanliness_scores) / len(cleanliness_scores))
            
            categories = [r["cleanliness_category"] for r in fully_successful if "cleanliness_category" in r]
            category_counts = {}
            for category in categories:
                category_counts[category] = category_counts.get(category, 0) + 1
            
            dominant_category = None
            if category_counts:
                dominant_category = max(category_counts.keys(), key=lambda x: category_counts[x])
            
            # Detect vehicle sides
            detected_sides = []
            side_detection_confidence = {}
            for r in fully_successful:
                if r.get("detected_vehicle_side") != "unknown":
                    detected_sides.append(r["detected_vehicle_side"])
                    side_detection_confidence[r["detected_vehicle_side"]] = r.get("detection_confidence", 0)
            
            vehicle_condition_summary = {
                "overall_cleanliness": dominant_category,
                "cleanliness_distribution": category_counts,
                "detected_vehicle_sides": list(set(detected_sides)),
                "side_detection_confidence": side_detection_confidence,
                "sides_analyzed": len(fully_successful),
                "sides_with_validation_errors": len(validation_failed),
                "sides_failed_processing": len(failed),
                "total_detections": sum(r.get("total_detections", 0) for r in fully_successful),
                "inference_method": "local"
            }
        
        # Generate assessment message
        inspection_status = "complete" if len(fully_successful) == IMAGE_CONFIG["required_files"] else "partial"
        
        if overall_cleanliness_score >= 90:
            assessment_message = "Excellent! Your vehicle is very clean and ready for rides."
        elif overall_cleanliness_score >= 80:
            assessment_message = "Great! Your vehicle is clean and meets quality standards."
        elif overall_cleanliness_score >= 70:
            assessment_message = "Good condition with minor cleaning needed."
        elif overall_cleanliness_score >= 50:
            assessment_message = "Vehicle needs cleaning before providing rides."
        else:
            assessment_message = "Vehicle requires thorough cleaning before operating."
        
        # Generate recommendations based on cleanliness categories
        recommendations = []
        
        dominant_category = vehicle_condition_summary.get("overall_cleanliness")
        
        if dominant_category == "very dirty":
            recommendations.append("Schedule a thorough professional cleaning")
            recommendations.append("Consider deep cleaning interior and exterior")
        elif dominant_category == "dirty":
            recommendations.append("Clean your vehicle before providing rides")
            recommendations.append("Pay attention to heavily soiled areas")
        elif dominant_category == "slightly dirty":
            recommendations.append("Quick wash and interior cleaning recommended")
        elif dominant_category == "clean":
            recommendations.append("Minor touch-ups to maintain excellent condition")
        elif dominant_category == "very clean":
            recommendations.append("Excellent! Your vehicle is ready for premium rides")
        
        if len(fully_successful) < IMAGE_CONFIG["required_files"]:
            recommendations.append("Retake failed photos for complete assessment")
        
        # Add specific recommendations for validation failures
        for result in validation_failed:
            if result.get("validation_error"):
                recommendations.append(result["validation_error"]["suggestion"])
        
        if failed:
            recommendations.append(f"Reprocess {len(failed)} failed images")
        
        if all_warnings:
            recommendations.append("Photo quality could be improved for more accurate assessment")
        
        # Add vehicle side detection recommendations
        detected_sides = vehicle_condition_summary.get("detected_vehicle_sides", [])
        if len(detected_sides) < len(fully_successful):
            recommendations.append("Some vehicle sides were not clearly detected - consider retaking photos")
        
        # Prepare response
        response_data = {
            "inspection_id": hash(str([f.filename for f in files])) % 10000000,
            "inspection_status": inspection_status,
            "inference_method": "local",
            "results": results,  # Legacy format for backward compatibility
            "side_results": side_results,  # New structured format for each side
            "validation": {
                "files_validated": len(files),
                "files_passed": len([v for v in validation_results if v["is_valid"]]),
                "warnings": list(set(all_warnings)),  # Remove duplicates
                "duplicates_detected": len(duplicates)
            },
            "summary": {
                "total_images_processed": len(files),
                "successful_predictions": len(fully_successful),
                "validation_failures": len(validation_failed),
                "processing_failures": len(failed),
                "overall_cleanliness_score": overall_cleanliness_score,
                "assessment_message": assessment_message,
                "vehicle_condition": vehicle_condition_summary
            },
            "recommendations": recommendations,
            "metadata": request_metadata,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Local inspection complete. Cleanliness score: {overall_cleanliness_score}%, Status: {inspection_status}")
        return response_data
        
    except ValidationError as ve:
        logger.error(f"Validation error: {ve.message}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": ve.message,
                "details": ve.details,
                "error_type": "validation_error",
                "inference_method": "local"
            }
        )
    
    except Exception as e:
        logger.error(f"Critical local inspection error: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Vehicle inspection failed",
                "details": "A critical error occurred during local processing",
                "technical_details": str(e),
                "error_type": "internal_error",
                "inference_method": "local"
            }
        )

@app.post("/inspect-vehicle")
async def inspect_vehicle(
    files: List[UploadFile] = File(...),
    metadata: Optional[str] = Form(None)
):
    """
    Enhanced vehicle inspection with comprehensive validation
    """
    try:
        # Parse metadata if provided
        request_metadata = {}
        if metadata:
            try:
                request_metadata = json.loads(metadata)
            except json.JSONDecodeError:
                logger.warning("Invalid metadata JSON provided")
        
        # Basic count validation
        if len(files) == 0:
            raise ValidationError(
                "No files uploaded",
                "Please upload at least one vehicle photo to continue"
            )
        
        if len(files) > IMAGE_CONFIG["max_files"]:
            raise ValidationError(
                f"Too many files uploaded ({len(files)})",
                f"Maximum {IMAGE_CONFIG['max_files']} files allowed"
            )
        
        if len(files) < IMAGE_CONFIG["required_files"]:
            missing_count = IMAGE_CONFIG["required_files"] - len(files)
            missing_sides = VEHICLE_SIDES[len(files):]
            raise ValidationError(
                f"Incomplete upload ({len(files)}/{IMAGE_CONFIG['required_files']} files)",
                f"Please upload photos for: {', '.join(missing_sides)}"
            )
        
        logger.info(f"Starting validation for {len(files)} files")
        
        # Validate each file
        validation_results = []
        all_warnings = []
        
        for i, file in enumerate(files):
            side_name = VEHICLE_SIDES[i] if i < len(VEHICLE_SIDES) else f"side_{i+1}"
            
            validation = validate_image_file(file)
            validation_results.append({
                "file_index": i,
                "filename": file.filename,
                "side": side_name,
                **validation
            })
            
            if not validation["is_valid"]:
                error_details = "; ".join(validation["errors"])
                raise ValidationError(
                    f"Invalid image for {side_name}",
                    f"{file.filename}: {error_details}"
                )
            
            all_warnings.extend(validation["warnings"])
        
        # Check for duplicate images
        duplicates = detect_duplicate_images(files)
        if duplicates:
            duplicate_info = ", ".join([
                f"{dup['filename']} is duplicate of {dup['duplicate_of_filename']}"
                for dup in duplicates
            ])
            raise ValidationError(
                "Duplicate images detected",
                f"Please use different photos for each vehicle side: {duplicate_info}"
            )
        
        logger.info("All files passed validation, proceeding with ML analysis")
        
        # Process images through ML service
        results = []
        for i, file in enumerate(files):
            side_name = VEHICLE_SIDES[i]
            result = await process_single_image(file, side_name)
            results.append(result)
        
        # Analyze results
        successful = [r for r in results if "error" not in r]
        failed = [r for r in results if "error" in r]
        
        # Separate validation errors from processing errors
        validation_failed = [r for r in successful if r.get("validation_error")]
        fully_successful = [r for r in successful if not r.get("validation_error")]
        
        # If any ML processing failed, include validation warnings
        if failed:
            logger.warning(f"{len(failed)} files failed ML processing")
        if validation_failed:
            logger.warning(f"{len(validation_failed)} files failed side validation")
        
        # Create structured response for each vehicle side
        side_results = {}
        for result in results:
            side = result["side"]
            side_results[side] = {
                "filename": result["filename"],
                "processing_status": "success" if "error" not in result else "failed",
                "validation_status": "passed" if not result.get("validation_error") else "failed",
                "cleanliness_category": result.get("cleanliness_category"),
                "cleanliness_score": result.get("cleanliness_score"),
                "detected_vehicle_side": result.get("detected_vehicle_side"),
                "detection_confidence": result.get("detection_confidence", 0),
                "total_detections": result.get("total_detections", 0),
                "validation_error": result.get("validation_error"),
                "processing_error": result.get("error")
            }
        
        # Calculate overall metrics (only from fully successful results)
        overall_cleanliness_score = 0
        vehicle_condition_summary = {}
        
        if fully_successful:
            cleanliness_scores = [r["cleanliness_score"] for r in fully_successful if "cleanliness_score" in r]
            if cleanliness_scores:
                overall_cleanliness_score = round(sum(cleanliness_scores) / len(cleanliness_scores))
            
            categories = [r["cleanliness_category"] for r in fully_successful if "cleanliness_category" in r]
            category_counts = {}
            for category in categories:
                category_counts[category] = category_counts.get(category, 0) + 1
            
            dominant_category = None
            if category_counts:
                dominant_category = max(category_counts.keys(), key=lambda x: category_counts[x])
            
            # Detect vehicle sides
            detected_sides = []
            side_detection_confidence = {}
            for r in fully_successful:
                if r.get("detected_vehicle_side") != "unknown":
                    detected_sides.append(r["detected_vehicle_side"])
                    side_detection_confidence[r["detected_vehicle_side"]] = r.get("detection_confidence", 0)
            
            vehicle_condition_summary = {
                "overall_cleanliness": dominant_category,
                "cleanliness_distribution": category_counts,
                "detected_vehicle_sides": list(set(detected_sides)),
                "side_detection_confidence": side_detection_confidence,
                "sides_analyzed": len(fully_successful),
                "sides_with_validation_errors": len(validation_failed),
                "sides_failed_processing": len(failed),
                "total_detections": sum(r.get("total_detections", 0) for r in fully_successful)
            }
        
        # Generate assessment message
        inspection_status = "complete" if len(fully_successful) == IMAGE_CONFIG["required_files"] else "partial"
        
        if overall_cleanliness_score >= 90:
            assessment_message = "Excellent! Your vehicle is very clean and ready for rides."
        elif overall_cleanliness_score >= 80:
            assessment_message = "Great! Your vehicle is clean and meets quality standards."
        elif overall_cleanliness_score >= 70:
            assessment_message = "Good condition with minor cleaning needed."
        elif overall_cleanliness_score >= 50:
            assessment_message = "Vehicle needs cleaning before providing rides."
        else:
            assessment_message = "Vehicle requires thorough cleaning before operating."
        
        # Generate recommendations based on cleanliness categories
        recommendations = []
        
        dominant_category = vehicle_condition_summary.get("overall_cleanliness")
        
        if dominant_category == "very dirty":
            recommendations.append("Schedule a thorough professional cleaning")
            recommendations.append("Consider deep cleaning interior and exterior")
        elif dominant_category == "dirty":
            recommendations.append("Clean your vehicle before providing rides")
            recommendations.append("Pay attention to heavily soiled areas")
        elif dominant_category == "slightly dirty":
            recommendations.append("Quick wash and interior cleaning recommended")
        elif dominant_category == "clean":
            recommendations.append("Minor touch-ups to maintain excellent condition")
        elif dominant_category == "very clean":
            recommendations.append("Excellent! Your vehicle is ready for premium rides")
        
        if len(fully_successful) < IMAGE_CONFIG["required_files"]:
            recommendations.append("Retake failed photos for complete assessment")
        
        # Add specific recommendations for validation failures
        for result in validation_failed:
            if result.get("validation_error"):
                recommendations.append(result["validation_error"]["suggestion"])
        
        if failed:
            recommendations.append(f"Reprocess {len(failed)} failed images")
        
        if all_warnings:
            recommendations.append("Photo quality could be improved for more accurate assessment")
        
        # Add vehicle side detection recommendations
        detected_sides = vehicle_condition_summary.get("detected_vehicle_sides", [])
        if len(detected_sides) < len(fully_successful):
            recommendations.append("Some vehicle sides were not clearly detected - consider retaking photos")
        
        # Prepare response
        response_data = {
            "inspection_id": hash(str([f.filename for f in files])) % 10000000,
            "inspection_status": inspection_status,
            "results": results,  # Legacy format for backward compatibility
            "side_results": side_results,  # New structured format for each side
            "validation": {
                "files_validated": len(files),
                "files_passed": len([v for v in validation_results if v["is_valid"]]),
                "warnings": list(set(all_warnings)),  # Remove duplicates
                "duplicates_detected": len(duplicates)
            },
            "summary": {
                "total_images_processed": len(files),
                "successful_predictions": len(fully_successful),
                "validation_failures": len(validation_failed),
                "processing_failures": len(failed),
                "overall_cleanliness_score": overall_cleanliness_score,
                "assessment_message": assessment_message,
                "vehicle_condition": vehicle_condition_summary
            },
            "recommendations": recommendations,
            "metadata": request_metadata,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Inspection complete. Cleanliness score: {overall_cleanliness_score}%, Status: {inspection_status}")
        return response_data
        
    except ValidationError as ve:
        logger.error(f"Validation error: {ve.message}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": ve.message,
                "details": ve.details,
                "error_type": "validation_error"
            }
        )
    
    except Exception as e:
        logger.error(f"Critical inspection error: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Vehicle inspection failed",
                "details": "A critical error occurred during processing",
                "technical_details": str(e),
                "error_type": "internal_error"
            }
        )

# ------------------------
# Error handlers
# ------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unexpected error: {exc}")
    return {
        "error": "Internal server error",
        "message": "An unexpected error occurred. Please try again.",
        "status_code": 500,
        "error_type": "global_exception"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)