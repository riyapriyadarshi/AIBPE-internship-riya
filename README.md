# AIBPE 

A comprehensive face analysis system that integrates multiple AI capabilities including face detection, recognition, demographics prediction, skin tone classification, and anomaly detection (acne and pigmentation).

## 🎯 Project Overview

AIBPE is an end-to-end face analysis pipeline that combines traditional computer vision techniques with deep learning models to provide comprehensive facial analysis. The system can detect faces, recognize identities, predict demographics (age, gender, ethnicity), classify skin tone, and detect skin anomalies like acne and pigmentation.

## ✨ Features

### Core Capabilities

- **Face Detection**: Haar Cascade-based face detection with bounding box extraction
- **Face Recognition**: Identity matching using deep learning embeddings and cosine similarity
- **Demographics Analysis**: Age, gender, and ethnicity prediction using MultiHeadResNet
- **Skin Tone Classification**: LAB color space-based skin tone categorization
- **Anomaly Detection**: Acne and pigmentation spot detection using image processing
- **Complete Pipeline**: Integrated end-to-end analysis combining all features

### API Support

- RESTful API built with FastAPI
- Interactive API documentation (Swagger UI)
- Multiple analysis endpoints
- File upload support for image analysis

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)
- [Week-by-Week Progress](#week-by-week-progress)
- [Requirements](#requirements)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (optional, for cloning the repository)

### Setup Steps

1. **Clone the repository** (or navigate to project directory):
   ```bash
   cd AIBPE-internship
   ```

2. **Create and activate virtual environment**:
   ```powershell
   # Windows PowerShell
   python -m venv aibpeenv
   .\aibpeenv\Scripts\Activate.ps1
   
   # Linux/Mac
   python -m venv aibpeenv
   source aibpeenv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download models** (if not already present):
   - Demographics model: `models/fairface/fairface_model.pt`
   - Face embeddings database: `models/face_embeddings.json`
   - YOLOv8 model: Automatically downloaded on first use

## ⚡ Quick Start

### Running the Complete Pipeline (Week 6)

```powershell
# Activate virtual environment
.\aibpeenv\Scripts\Activate.ps1

# Run with default sample image
python main.py

# Run with specific image
python main.py --image path/to/your/image.jpg
```

### Running the API Server

```powershell
# Activate virtual environment
.\aibpeenv\Scripts\Activate.ps1

# Start the API server
python api.py
```

The API will be available at `http://localhost:8000`

Access interactive documentation at: `http://localhost:8000/docs`

## 📖 Usage

### Command Line Interface

#### 1. Face Detection
```bash
python facedetector.py --input image.jpg --out outputs/ --save-crops
```

#### 2. Generate Face Embeddings
```bash
python generate_embeddings.py --data_dir path/to/images --out models/face_embeddings.json
```

#### 3. Face Recognition
```bash
python recognize_face.py --query image.jpg --db models/face_embeddings.json --threshold 0.55
```

#### 4. Demographics Detection
```bash
python demographics_detector.py --input image.jpg
```

#### 5. Skin Tone Classification
```bash
python skin_tone_classifier.py --input image.jpg
```

#### 6. Anomaly Detection (Acne & Pigmentation)
```bash
python anomaly_detector.py --input image.jpg --output outputs/ --confidence 0.25
```

#### 7. Complete Pipeline (Week 6)
```bash
python main.py --image image.jpg --output outputs/
```

### Python API

```python
from main import run_full_pipeline

# Run complete analysis
results = run_full_pipeline("image.jpg", output_dir="outputs")
print(f"Faces detected: {len(results['face_detection'].get('detections', []))}")
print(f"Identity: {results['face_recognition'].get('identity')}")
print(f"Demographics: {results['demographics']}")
print(f"Skin tone: {results['skin_tone'].get('skin_tone_category')}")
print(f"Anomalies: {results['anomaly_detection'].get('total_detections')}")
```

## 🌐 API Documentation

### Base URL
```
http://localhost:8000
```

### Available Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information and endpoint list |
| `/analyze/demographics` | POST | Analyze age, gender, ethnicity |
| `/analyze/skintone` | POST | Classify skin tone |
| `/analyze/anomalies` | POST | Detect acne and pigmentation |
| `/analyze/full` | POST | Combined analysis (3 steps) |
| `/analyze/pipeline` | POST | Complete Week 6 pipeline (6 steps) |
| `/docs` | GET | Interactive API documentation (Swagger UI) |
| `/redoc` | GET | Alternative API documentation |

### Example API Request

```python
import requests

# Anomaly detection (acne & pigmentation)
with open("sample.jpg", "rb") as f:
    files = {"file": ("sample.jpg", f, "image/jpeg")}
    response = requests.post(
        "http://localhost:8000/analyze/anomalies",
        files=files,
        params={"confidence": 0.25}
    )
    result = response.json()
    print(result)
```

### Using curl

```bash
# Test API
curl http://localhost:8000/

# Anomaly detection
curl -X POST "http://localhost:8000/analyze/anomalies" \
  -F "file=@sample.jpg" \
  -F "confidence=0.25"
```

### Interactive Documentation

Visit `http://localhost:8000/docs` in your browser to access Swagger UI, where you can:
- Test all endpoints interactively
- Upload images directly
- View request/response schemas
- See example requests

## 📁 Project Structure

```
AIBPE-internship/
├── main.py                      # Week 6: Complete integrated pipeline
├── api.py                       # FastAPI server for all endpoints
│
├── facedetector.py              # Week 1: Face detection using Haar Cascade
├── embedding_model.py           # Week 2: Face embedding model (ResNet50)
├── generate_embeddings.py       # Week 2: Generate embeddings database
├── recognize_face.py            # Week 2: Face recognition using embeddings
├── skin_tone_classifier.py      # Week 3: Skin tone classification
├── anomaly_detector.py          # Week 4: Acne & pigmentation detection
├── demographics_detector.py     # Week 5: Demographics prediction
│
├── models/                      # Model files and databases
│   ├── fairface/               # Demographics model
│   │   ├── fairface_model.pt
│   │   └── fairface_label_dict.json
│   ├── face_embeddings.json    # Face recognition database
│   └── yolo/                   # YOLOv8 models (auto-downloaded)
│
├── outputs/                     # Analysis results and outputs
├── kj/                         # Sample images (person kj)
├── bh/                         # Sample images (person bh)
├── ma/                         # Sample images (person ma)
│
├── requirements.txt             # Python dependencies
├── README.md                   # This file
└── test_*.py                   # Test scripts for API and modules
```

## 📅 Week-by-Week Progress

### Week 1: Face Detection
- ✅ Implemented Haar Cascade face detection
- ✅ Face cropping and saving functionality
- ✅ Output metadata in JSON format

**File:** `facedetector.py`

### Week 2: Face Recognition
- ✅ ResNet50-based embedding extraction
- ✅ Embedding database generation
- ✅ Face recognition using cosine similarity
- ✅ Identity matching with confidence scores

**Files:** `embedding_model.py`, `generate_embeddings.py`, `recognize_face.py`

### Week 3: Skin Tone Classification
- ✅ LAB color space conversion
- ✅ Average L*, a*, b* value computation
- ✅ Skin tone category mapping
- ✅ Classification into 6 categories (Very Light to Dark)

**File:** `skin_tone_classifier.py`

### Week 4: Anomaly Detection
- ✅ Face region detection and cropping
- ✅ Acne detection (red/inflamed spots)
- ✅ Pigmentation detection (dark spots)
- ✅ Bounding box visualization
- ✅ JSON output with detection results

**File:** `anomaly_detector.py`

### Week 5: Demographics Detection
- ✅ MultiHeadResNet model for multi-task learning
- ✅ Age, gender, and ethnicity prediction
- ✅ Model loading and inference
- ✅ Face detection integration

**File:** `demographics_detector.py`, `api.py`

### Week 6: Final Integration
- ✅ Complete end-to-end pipeline
- ✅ Integration of all modules
- ✅ Unified output format
- ✅ Command-line interface
- ✅ API endpoint for complete pipeline

**File:** `main.py`

## 📦 Requirements

Key dependencies (see `requirements.txt` for complete list):

- `opencv-python` - Computer vision and image processing
- `torch` / `pytorch` - Deep learning framework
- `torchvision` - Vision utilities for PyTorch
- `ultralytics` - YOLOv8 for object detection
- `fastapi` - Web API framework
- `uvicorn` - ASGI server
- `pillow` - Image processing
- `numpy` - Numerical computing
- `requests` - HTTP library (for API testing)

Install all dependencies:
```bash
pip install -r requirements.txt
```

## 💡 Examples

### Example 1: Run Complete Pipeline

```powershell
python main.py --image sample.jpg
```

**Output:** Creates `outputs/sample_full_analysis.json` with:
- Face detection results
- Face embeddings
- Face recognition (identity match)
- Demographics (age, gender, ethnicity)
- Skin tone classification
- Anomaly detection (acne & pigmentation counts)

### Example 2: API Request for Anomaly Detection

```python
import requests

with open("sample.jpg", "rb") as f:
    files = {"file": ("sample.jpg", f, "image/jpeg")}
    response = requests.post(
        "http://localhost:8000/analyze/anomalies",
        files=files,
        params={"confidence": 0.25}
    )
    result = response.json()
    
    print(f"Acne spots: {result['result']['acne_count']}")
    print(f"Pigmentation spots: {result['result']['pigmentation_count']}")
```

### Example 3: Generate Face Embeddings Database

```powershell
# Structure: data_dir/
#   person1/
#     image1.jpg
#     image2.jpg
#   person2/
#     image1.jpg

python generate_embeddings.py --data_dir path/to/images --out models/face_embeddings.json
```

## 🔧 Troubleshooting

### Common Issues

**1. ModuleNotFoundError**
- **Solution:** Ensure virtual environment is activated and dependencies are installed
```powershell
.\aibpeenv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**2. Model not found errors**
- **Solution:** Ensure model files exist in `models/` directory
- Demographics model: `models/fairface/fairface_model.pt`
- Face embeddings: `models/face_embeddings.json`
- YOLOv8: Auto-downloaded on first use

**3. API not responding**
- **Solution:** Check if server is running
```powershell
# Check if running
curl http://localhost:8000/

# Restart server
python api.py
```

**4. Face not detected**
- **Solution:** Ensure image has clear, front-facing face
- Try adjusting image quality or lighting
- Check if face is large enough in the image

**5. Low recognition confidence**
- **Solution:** Face might not be in database
- Generate embeddings for the person first
- Adjust threshold in `recognize_face.py`

### Getting Help

- Check the interactive API docs: `http://localhost:8000/docs`
- Review test scripts: `test_*.py`
- Check output files in `outputs/` directory for examples

## 📝 Output Format

The complete pipeline generates JSON output with the following structure:

```json
{
  "input_image": "sample.jpg",
  "face_detection": {
    "detections": [...]
  },
  "embedding": [2048-dimensional vector],
  "face_recognition": {
    "identity": "person_name",
    "score": 0.85,
    "match_file": "path/to/match.jpg"
  },
  "demographics": {
    "gender": "Female",
    "age": "20-29",
    "ethnicity": "East Asian",
    "age_confidence": 0.51
  },
  "skin_tone": {
    "skin_tone_category": "Medium Light",
    "lab_values": {"L": 59.67, "a": 19.84, "b": 18.57}
  },
  "anomaly_detection": {
    "face_detected": true,
    "acne_count": 3,
    "pigmentation_count": 4,
    "total_detections": 7,
    "detections": [...]
  }
}
```

## 🧪 Testing

Test scripts are available for various components:

```powershell
# Test API endpoints
python test_api.py

# Test anomaly detection API
python test_anomaly_api.py

# Test pipeline API
python test_pipeline_api.py

# Test individual modules (examples)
python anomaly_detector.py --input sample.jpg --verbose
python demographics_detector.py --input sample.jpg
```

## 📄 License

This project is developed as part of an internship program.

## 🙏 Acknowledgments

- Haar Cascade classifiers (OpenCV)
- ResNet50 architecture (PyTorch/Torchvision)
- YOLOv8 (Ultralytics)
- FairFace dataset and model architecture
- FastAPI framework

## 📧 Contact

For questions or issues related to this project, please refer to the project documentation or contact the development team.

---

**Last Updated:** Week 6 - Final Integration Complete

