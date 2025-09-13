from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

USE_DUMMY_PREDICTIONS = True

if not USE_DUMMY_PREDICTIONS:
    import joblib
    model = joblib.load("app/model.joblib")

class_names = np.array(["clean", "dirty", "scratchless", "scratched"])

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {
        "message": "inDrive AI Image Analysis API",
        "docs_url": "/docs"
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_status": "dummy" if USE_DUMMY_PREDICTIONS else "loaded",
        "class_labels": class_names.tolist(),
        "version": "1.0.0"
    }

def preprocess_image(image: Image.Image, model_type="sklearn"):
    img_array = np.array(image) / 255.0
    
    if model_type == "sklearn":
        return img_array.reshape(1, -1)
    
    elif model_type == "cnn":
        return np.expand_dims(img_array, axis=0)
    
    else:
        raise ValueError("Unknown model type")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predicts the class of an uploaded image with confidence scores.
    """
    try:
        if not file or not file.filename:
            return {"error": "No file uploaded or missing filename."}
        
        if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
            return {"error": "Invalid file type. Only JPG and PNG are supported."}

        MAX_FILE_SIZE = 5 * 1024 * 1024
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            return {"error": "File too large. Max size is 5 MB."}

        logger.info(f"Processing image: {file.filename}, size: {len(contents)} bytes")
        
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((224, 224))
        
        if USE_DUMMY_PREDICTIONS:
            np.random.seed(hash(file.filename) % 2**32)
            dummy_probs = np.random.dirichlet([2, 1, 1, 1])
            prediction_idx = np.argmax(dummy_probs)
            
            class_name = class_names[prediction_idx]
            confidence = float(dummy_probs[prediction_idx])
            
            logger.info(f"DUMMY Prediction: {class_name}, Confidence: {confidence:.3f}")
            
            return {
                "predicted_class": class_name,
                "confidence": confidence,
                "all_scores": dict(zip(class_names, dummy_probs.tolist())),
                "note": "Using dummy predictions - replace with real model"
            }
        
        else:
            img_array = preprocess_image(image, model_type="sklearn")
            
            prediction = model.predict(img_array)
            prediction_proba = model.predict_proba(img_array)
            
            class_name = class_names[prediction[0]]
            confidence = float(np.max(prediction_proba))
            
            logger.info(f"Real Prediction: {class_name}, Confidence: {confidence:.3f}")
            
            return {
                "predicted_class": class_name,
                "confidence": confidence,
                "all_scores": dict(zip(class_names, prediction_proba[0].tolist()))
            }
        
    except Exception as e:
        logger.error(f"Error processing image {file.filename if file else 'unknown'}: {str(e)}")
        return {
            "error": "Image processing failed",
            "details": str(e),
            "hint": "Ensure the uploaded file is a valid image (jpg/png)."
        }
    
@app.post("/trust-score")
async def trust_score(file: UploadFile = File(...)):
    result = await predict(file)
    
    if "predicted_class" not in result:
        return result
    
    confidence = float(result.get("confidence", 0.0))
    trust_score = round(confidence * 100, 2)

    logger.info(f"Trust Score for {file.filename}: {trust_score}% ({result['predicted_class']})")
    
    return {
        "trust_score": trust_score,
        "predicted_class": result["predicted_class"],
        "explanation": (
            f"The photo was classified as '{result['predicted_class']}' "
            f"with {trust_score}% confidence."
        )
    }