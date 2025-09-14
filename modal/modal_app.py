import modal
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from ultralytics import YOLO
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io

# -------------------------
# Modal app & image setup
# -------------------------
app = modal.App("inDrive-vehicle-inspector")

# Base image with required Python libs including PyTorch and YOLO
image = (
    modal.Image.debian_slim()
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "libsm6", "libxext6", "libxrender-dev", "libgomp1")
    .pip_install(
        "fastapi",
        "uvicorn[standard]",
        "pillow",
        "numpy",
        "torch",
        "torchvision",
        "ultralytics",
        "python-multipart"
    ).add_local_file("./best_car_parts.pt", "/root/best_car_parts.pt").add_local_file(
        "./efficientnet_binary_clean_dirty.pth", "/root/efficientnet_binary_clean_dirty.pth")
)

# -------------------------
# FastAPI app inside Modal
# -------------------------
web_app = FastAPI(
    title="inDrive ML Service",
    description="Dual model inference: Car parts detection + Clean/Dirty classification",
    version="2.0.0"
)

# Global variables to hold loaded models
car_parts_model = None
clean_dirty_model = None

# Image preprocessing transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


def load_models():
    """Load both models: YOLO for car parts and EfficientNet for clean/dirty"""
    global car_parts_model, clean_dirty_model

    try:
        # Load YOLO car parts model
        car_parts_model = YOLO("/root/best_car_parts.pt")

        # Load clean/dirty model - handle both state dict and full model
        try:
            clean_dirty_model = torch.load("/root/efficientnet_binary_clean_dirty.pth", map_location="cuda")
            if hasattr(clean_dirty_model, 'eval'):
                clean_dirty_model.eval()
            else:
                # It's a state dict, recreate the model
                from torchvision.models import efficientnet_b0
                model = efficientnet_b0(weights=None)
                
                # Handle EfficientNet classifier structure
                # The saved model has classifier.1.weight/bias with shape [2, 1280], indicating 2 classes
                # EfficientNet-B0 default structure: classifier = Sequential(Dropout, Linear)
                model.classifier = torch.nn.Sequential(
                    torch.nn.Dropout(p=0.2, inplace=True),
                    torch.nn.Linear(1280, 2)  # 2 classes: clean, dirty
                )
                
                model.load_state_dict(clean_dirty_model)
                clean_dirty_model = model.cuda()
                clean_dirty_model.eval()
        except Exception as model_error:
            print(f"Model loading error: {model_error}")
            raise

        print("✅ Both models loaded successfully")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        raise


def predict_car_parts(img: Image.Image):
    """Predict car parts using YOLO model and map to vehicle sides"""
    try:
        results = car_parts_model(img)
        detections = []

        # Use the model's actual class names instead of hardcoded mapping
        # Model names: ['front', 'left', 'rear', 'right'] (indices 0,1,2,3)

        for r in results:
            for b in r.boxes:
                cls = int(b.cls)
                conf = float(b.conf)
                # Get bounding box coordinates
                xyxy = b.xyxy[0].tolist()  # [x1, y1, x2, y2]

                # Get the actual class name from the model
                vehicle_side = car_parts_model.names.get(cls, f"unknown_part_{cls}")

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


def predict_clean_dirty(img: Image.Image):
    """Predict clean/dirty using the EfficientNet model"""
    try:
        img_rgb = img.convert("RGB")
        x = transform(img_rgb).unsqueeze(0).cuda()

        with torch.no_grad():
            out = clean_dirty_model(x)
            probs = torch.softmax(out, dim=1)
            dirty_prob = probs[0][1].item()  # Get probability of dirty class (index 1)

        return {"dirty_prob": dirty_prob}
    except Exception as e:
        return {"error": f"Clean/dirty prediction failed: {str(e)}"}


@web_app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts 1 image → returns predictions from both models.
    """
    try:
        # Load models on first request
        if car_parts_model is None or clean_dirty_model is None:
            load_models()

        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")

        # Get predictions from both models
        car_parts_result = predict_car_parts(img)
        clean_dirty_result = predict_clean_dirty(img)

        return {
            "filename": file.filename,
            "car_parts_model": car_parts_result,
            "clean_dirty_model": clean_dirty_result,
            "status": "success"
        }

    except Exception as e:
        return {
            "error": "Failed to process image",
            "details": str(e),
            "filename": file.filename or "unknown"
        }


@web_app.get("/health")
def health_check():
    """Health check for ML service"""
    return {
        "status": "healthy",
        "service": "indrive-ml-dual-model",
        "models": {
            "car_parts_loaded": car_parts_model is not None,
            "clean_dirty_loaded": clean_dirty_model is not None
        }
    }


# -------------------------
# Expose FastAPI via Modal
# -------------------------
@app.function(
    image=image,
    gpu=modal.gpu.T4(),
)
@modal.asgi_app()
def fastapi_app():
    return web_app
