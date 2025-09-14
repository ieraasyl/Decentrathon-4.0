# Decentrathon 4.0 - Vehicle Inspector 🚗

AI-powered vehicle condition assessment for inDrive. Upload 4 photos of your car and get instant quality analysis with trust scoring.

## ⚡ Quick Start

**Prerequisites:** Docker & Docker Compose installed

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd Decentrathon-4.0
   ```

2. **Start the application**
   ```bash
   docker-compose up
   ```

3. **Open in browser**
   - Frontend: http://localhost:5173
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

That's it! 🎉

## 🏗️ Architecture

- **Frontend**: React 19 + Vite + TailwindCSS
- **Backend**: FastAPI + Python 3.11
- **ML Service**: Modal.com (serverless ML inference)
- **Database**: File-based (model.joblib)

## 📸 Features

- Upload 4 vehicle photos (front, back, left, right)
- AI condition assessment (clean/dirty, scratched/scratchless)
- Real-time image validation
- Trust scoring system (0-100%)
- Comprehensive error handling
- Mobile-responsive UI

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

### Environment Variables

Copy `.env.example` to `.env` and adjust if needed:
```bash
cp .env.example .env
```

## 🚀 Team ACM+1
