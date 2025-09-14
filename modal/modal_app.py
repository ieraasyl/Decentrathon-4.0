import modal
import numpy as np
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io

# -------------------------
# Modal app & image setup
# -------------------------
app = modal.App("inDrive-vehicle-inspector")

# Base image with required Python libs
image = (
    modal.Image.debian_slim()
    .pip_install("fastapi", "uvicorn[standard]", "pillow", "numpy", "scikit-learn", "joblib")
)

# -------------------------
# FastAPI app inside Modal
# -------------------------
web_app = FastAPI(
    title="inDrive ML Service",
    description="Handles image preprocessing and ML inference for vehicle condition",
    version="1.0.0"
)

CLASS_NAMES = ["clean", "dirty", "scratchless", "scratched"]

def preprocess(img: Image.Image):
    """
    Resize + normalize image.
    Replace this later with real CNN preprocessing.
    """
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    return arr.reshape(1, -1)

@web_app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts 1 image → returns ML prediction.
    """
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")

        # Example preprocessing call (currently unused)
        _ = preprocess(img)

        # Dummy prediction (replace with real model later)
        np.random.seed(hash(file.filename or "default") % 2**32)
        probs = np.random.dirichlet([2, 1, 1, 1])
        idx = np.argmax(probs)

        return {
            "predicted_class": CLASS_NAMES[idx],
            "confidence": float(probs[idx]),
            "trust_score": round(float(probs[idx]) * 100, 2),
            "all_scores": dict(zip(CLASS_NAMES, probs.tolist())),
        }

    except Exception as e:
        return {
            "error": "Failed to process image",
            "details": str(e),
            "hint": "Ensure the uploaded file is a valid JPG/PNG image."
        }

@web_app.get("/health")
def health_check():
    """Health check for ML service"""
    return {"status": "healthy", "service": "indrive-ml"}

# -------------------------
# Expose FastAPI via Modal
# -------------------------
@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    return web_app