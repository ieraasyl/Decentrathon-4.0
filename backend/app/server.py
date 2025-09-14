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
            
            # Validate ML service response
            required_fields = ["predicted_class", "confidence", "trust_score"]
            missing_fields = [field for field in required_fields if field not in prediction]
            
            if missing_fields:
                return {
                    "side": side_name,
                    "filename": file.filename,
                    "error": "Invalid ML response",
                    "details": f"Missing fields: {missing_fields}"
                }
            
            logger.info(f"Prediction for {side_name}: {prediction}")
            return {
                "side": side_name,
                "filename": file.filename,
                "processing_attempts": attempt + 1,
                **prediction
            }
            
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
    
    return {
        "status": "healthy",
        "ml_service_url": MODAL_ML_URL,
        "ml_service_status": ml_service_status,
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
        "supported_formats": ["JPG", "JPEG", "PNG", "WebP"]
    }

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
        
        # If any ML processing failed, include validation warnings
        if failed:
            logger.warning(f"{len(failed)} files failed ML processing")
        
        # Calculate overall metrics
        overall_trust_score = 0
        vehicle_condition_summary = {}
        
        if successful:
            trust_scores = [r["trust_score"] for r in successful if "trust_score" in r]
            if trust_scores:
                overall_trust_score = round(sum(trust_scores) / len(trust_scores), 2)
            
            conditions = [r["predicted_class"] for r in successful if "predicted_class" in r]
            condition_counts = {}
            for condition in conditions:
                condition_counts[condition] = condition_counts.get(condition, 0) + 1
            
            dominant_condition = None
            if condition_counts:
                dominant_condition = max(condition_counts.keys(), key=lambda x: condition_counts[x])
            
            vehicle_condition_summary = {
                "dominant_condition": dominant_condition,
                "condition_distribution": condition_counts,
                "sides_analyzed": len(successful),
                "sides_failed": len(failed)
            }
        
        # Generate assessment message
        inspection_status = "complete" if len(successful) == IMAGE_CONFIG["required_files"] else "partial"
        
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
        
        # Generate recommendations
        recommendations = []
        
        if overall_trust_score < 80:
            recommendations.append("Consider cleaning your vehicle before providing rides")
        
        if any(r.get("predicted_class") == "scratched" for r in successful):
            recommendations.append("Address visible scratches to improve passenger confidence")
        
        if len(successful) < IMAGE_CONFIG["required_files"]:
            recommendations.append("Retake failed photos for complete assessment")
        
        if overall_trust_score >= 90:
            recommendations.append("Excellent condition! You're ready for premium rides")
        
        if all_warnings:
            recommendations.append("Photo quality could be improved for more accurate assessment")
        
        # Prepare response
        response_data = {
            "inspection_id": hash(str([f.filename for f in files])) % 10000000,
            "inspection_status": inspection_status,
            "results": results,
            "validation": {
                "files_validated": len(files),
                "files_passed": len([v for v in validation_results if v["is_valid"]]),
                "warnings": list(set(all_warnings)),  # Remove duplicates
                "duplicates_detected": len(duplicates)
            },
            "summary": {
                "total_images_processed": len(files),
                "successful_predictions": len(successful),
                "failed_predictions": len(failed),
                "overall_trust_score": overall_trust_score,
                "assessment_message": assessment_message,
                "vehicle_condition": vehicle_condition_summary
            },
            "recommendations": recommendations,
            "metadata": request_metadata,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Inspection complete. Trust score: {overall_trust_score}%, Status: {inspection_status}")
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