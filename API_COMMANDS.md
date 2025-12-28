# 🚀 FastAPI Face Analysis Service - Complete Command Guide

## ⚡ Quick Start (All-in-One)

```powershell
# 1. Activate virtual environment and start server
.\aibpeenv\Scripts\Activate.ps1
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

**Then in a new terminal:**
```powershell
# 2. Test the API
.\aibpeenv\Scripts\Activate.ps1
python test_api.py bh/b3.jpeg
```

---

## 📝 Step-by-Step Commands

### Step 1: Start the API Server

```powershell
# Navigate to project directory (if not already there)
cd "C:\Users\riyap\OneDrive\Desktop\AIBPE-internship"

# Activate virtual environment
.\aibpeenv\Scripts\Activate.ps1

# Start FastAPI server
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

**Expected output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
[i] Loading demographics model on device: cpu
[i] Loaded full model object
[i] Demographics model loaded successfully
INFO:     Application startup complete.
```

---

### Step 2: Test Individual Endpoints

#### Test Root Endpoint
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/" -Method Get | ConvertTo-Json
```

#### Test Demographics Analysis
```powershell
$response = Invoke-RestMethod -Uri "http://localhost:8000/analyze/demographics" -Method Post -InFile "bh/b3.jpeg" -ContentType "multipart/form-data"
$response | ConvertTo-Json -Depth 10
```

**Expected result:**
```json
{
  "status": "success",
  "analysis": "demographics",
  "result": {
    "gender": "Male",
    "age": "30-39",
    "age_confidence": 0.2106,
    "age_alternatives": [
      {"age": "30-39", "confidence": 0.2106},
      {"age": "20-29", "confidence": 0.1984},
      {"age": "3-9", "confidence": 0.1565}
    ],
    "ethnicity": "Latino_Hispanic"
  }
}
```

#### Test Skin Tone Analysis
```powershell
$response = Invoke-RestMethod -Uri "http://localhost:8000/analyze/skintone" -Method Post -InFile "bh/b3.jpeg" -ContentType "multipart/form-data"
$response | ConvertTo-Json -Depth 10
```

**Expected result:**
```json
{
  "status": "success",
  "analysis": "skintone",
  "result": {
    "input_image": "...",
    "face_crop": "auto-detected",
    "lab_values": {
      "L": 45.23,
      "a": 12.45,
      "b": 18.67
    },
    "skin_tone_category": "Medium",
    "description": "Medium skin tone"
  }
}
```

#### Test Anomaly Detection
```powershell
$response = Invoke-RestMethod -Uri "http://localhost:8000/analyze/anomalies?confidence=0.25" -Method Post -InFile "bh/b3.jpeg" -ContentType "multipart/form-data"
$response | ConvertTo-Json -Depth 10
```

**Expected result:**
```json
{
  "status": "success",
  "analysis": "anomalies",
  "result": {
    "input_image": "...",
    "detections": [
      {
        "class": "person",
        "confidence": 0.92,
        "bbox": {...}
      }
    ],
    "total_detections": 1,
    "confidence_threshold": 0.25
  }
}
```

#### Test Full Analysis (All Three Combined)
```powershell
$response = Invoke-RestMethod -Uri "http://localhost:8000/analyze/full?confidence=0.25" -Method Post -InFile "bh/b3.jpeg" -ContentType "multipart/form-data"
$response | ConvertTo-Json -Depth 10
```

**Expected result:**
```json
{
  "status": "success",
  "analysis": "full",
  "results": {
    "demographics": {...},
    "skintone": {...},
    "anomalies": {...}
  },
  "errors": null
}
```

---

### Step 3: Test with Python Script

```powershell
.\aibpeenv\Scripts\Activate.ps1
python test_api.py bh/b3.jpeg
```

**Or test with other images:**
```powershell
python test_api.py kj/k1.jpeg
python test_api.py bh/b1.jpeg
python test_api.py bh/b2.jpeg
```

---

## 🌐 Access Interactive Documentation

Once the server is running, open your browser:

1. **Swagger UI (Interactive)**: http://localhost:8000/docs
   - Try endpoints directly in the browser
   - See request/response schemas
   - Upload files and test

2. **ReDoc (Readable)**: http://localhost:8000/redoc
   - Clean documentation format
   - Easy to read

---

## 🔧 Using Python Requests Library

```python
import requests

# Demographics
with open("bh/b3.jpeg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/analyze/demographics",
        files={"file": f}
    )
    print(response.json())

# Skin Tone
with open("bh/b3.jpeg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/analyze/skintone",
        files={"file": f}
    )
    print(response.json())

# Anomalies with confidence threshold
with open("bh/b3.jpeg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/analyze/anomalies",
        files={"file": f},
        params={"confidence": 0.25}
    )
    print(response.json())

# Full Analysis
with open("bh/b3.jpeg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/analyze/full",
        files={"file": f},
        params={"confidence": 0.25}
    )
    print(response.json())
```

---

## 📋 Complete Command Reference

### Start Server
```powershell
.\aibpeenv\Scripts\Activate.ps1
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

### Test All Endpoints (Automated)
```powershell
.\aibpeenv\Scripts\Activate.ps1
python test_api.py bh/b3.jpeg
```

### Test Individual Endpoints (PowerShell)
```powershell
# Demographics
Invoke-RestMethod -Uri "http://localhost:8000/analyze/demographics" -Method Post -InFile "bh/b3.jpeg" -ContentType "multipart/form-data"

# Skin Tone
Invoke-RestMethod -Uri "http://localhost:8000/analyze/skintone" -Method Post -InFile "bh/b3.jpeg" -ContentType "multipart/form-data"

# Anomalies
Invoke-RestMethod -Uri "http://localhost:8000/analyze/anomalies?confidence=0.25" -Method Post -InFile "bh/b3.jpeg" -ContentType "multipart/form-data"

# Full Analysis
Invoke-RestMethod -Uri "http://localhost:8000/analyze/full?confidence=0.25" -Method Post -InFile "bh/b3.jpeg" -ContentType "multipart/form-data"
```

### Stop Server
Press `Ctrl+C` in the terminal where the server is running.

---

## ⚠️ Troubleshooting

### Model Not Loading
If you see "Demographics model not loaded":
1. Make sure you restarted the server after the code was updated
2. Check that `models/fairface/fairface_model.pt` exists
3. Look at server startup logs for error messages

### Port Already in Use
If port 8000 is busy:
```powershell
uvicorn api:app --reload --host 0.0.0.0 --port 8001
```
Then update URLs to use port 8001.

### Import Errors
Make sure virtual environment is activated:
```powershell
.\aibpeenv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## ✅ Verification Checklist

- [ ] Server starts without errors
- [ ] Model loads successfully (check startup logs)
- [ ] Root endpoint returns API info
- [ ] Demographics endpoint works
- [ ] Skin tone endpoint works
- [ ] Anomalies endpoint works
- [ ] Full analysis endpoint works
- [ ] Interactive docs accessible at /docs

