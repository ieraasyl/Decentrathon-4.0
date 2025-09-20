# Decentrathon 4.0 - Vehicle Inspector 🚗

AI-powered vehicle condition assessment for inDrive. Upload 4 photos of your car and get instant quality analysis with trust scoring.

## ⚡ Quick Start

**Prerequisites:** 
- [Docker](https://docs.docker.com/get-docker/) installed on your machine
- Docker Compose (included with Docker Desktop)

1. **Clone the repository and download the models**
   ```bash
   git clone https://github.com/ieraasyl/Decentrathon-4.0
   cd Decentrathon-4.0
   ```

   Car masking model (PyTorch): [Link to the model for download](https://drive.google.com/file/d/144_IXtXueFzik6pzqhRvndIizZV-1Zfk/view?usp=drive_link)
   Damage detection model (ONNX): [Link to the onnx model](https://drive.google.com/file/d/1E_1HjlxkKZFoMLN_LLOEkoqXnzW2hGw6/view?usp=drive_link)
   Damage detection model (PyTorch): [Link to the pytorch model](https://drive.google.com/file/d/1WGvWEHgGZsKZrbwxfQ8K1lBoYO7XhkYK/view?usp=drive_link)

2. **Start the application**
   ```bash
   docker compose up
   ```

3. **Open in browser**
   - Frontend: http://localhost:5173
   - Backend API: http://localhost:8000
That's it! 🎉

## 🏗️ Architecture

- **Frontend**: React 19 + Vite + TailwindCSS
- **Backend**: FastAPI + Python 3.11 + PyTorch
- **ML Models**: 
  - **YOLO**: Vehicle side detection (`best_car_parts.pt`)
  - **EfficientNet**: Cleanliness classification (`efficientnet_binary_clean_dirty.pth`)
- **Deployment**: Modal.com (cloud) or Local Docker (development)
- **Validation**: Multi-layer image and ML validation pipeline

## 📸 Features

### 🚗 Vehicle Inspection
- **4-sided analysis**: Upload front, back, left, right vehicle photos
- **YOLO-powered detection**: Identifies vehicle parts and orientation
- **Cleanliness assessment**: 5-level scoring (very clean → very dirty) 
- **Side mismatch validation**: Prevents wrong image placement with confidence scores
- **Smart recommendations**: Prioritizes worst-case scenarios for cleaning

### 🛡️ Advanced Validation
- **Comprehensive image validation**: File type, size, dimensions, aspect ratio
- **Duplicate detection**: Prevents same image for multiple sides
- **Real-time feedback**: Loading states and success/error messages
- **Strict quality control**: Blocks analysis until images are correctly placed

### ⚙️ Deployment Options
- **Docker Compose**: Local ML models (YOLO + EfficientNet) for fast inference
- **Production**: Modal API for scalable cloud processing
- **Environment-based switching**: `USE_LOCAL_MODELS=true/false`

### 🎨 User Experience  
- **Mobile-first design**: Responsive UI with Tailwind CSS
- **Progressive validation**: Step-by-step guidance with helpful tips
- **Error recovery**: Clear messaging and suggestions for fixing issues
- **Professional scoring**: Color-coded results with actionable recommendations

## 🛠️ Development

### Manual Setup (Alternative)

**Backend:**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.server:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

### ML Model Setup (Local Development)

For local model inference, ensure model files are in the `models/` directory:

```bash
models/
├── best_car_parts.pt           # YOLO vehicle detection model
└── efficientnet_binary_clean_dirty.pth  # Cleanliness classification
```

**Docker Compose automatically:**
- Mounts `./models` to `/code/models` in backend container
- Sets `USE_LOCAL_MODELS=true` for local inference
- Loads models during startup (container fails if models can't load)

### Environment Variables

Key environment variables:
- `USE_LOCAL_MODELS=true` - Use local models (Docker Compose)
- `USE_LOCAL_MODELS=false` - Use Modal API (production)
- `MODAL_ML_URL` - Modal API endpoint for cloud inference

## 🔧 API Endpoints

- `GET /health` - System health check with model status
- `POST /inspect-vehicle` - Main vehicle inspection endpoint
- `POST /predict` - Single image prediction (local models)
- `GET /` - API information and capabilities

## 🚀 Team ACM+1
