# api.py
# Week 5: FastAPI Service for Face Analysis
#
# Endpoints:
# - /analyze/demographics - Analyze demographics (age, gender, ethnicity)
# - /analyze/skintone - Classify skin tone
# - /analyze/anomalies - Detect anomalies using YOLOv8
# - /analyze/full - Combined analysis (all three)
#
# Usage:
#   uvicorn api:app --reload
#   uvicorn api:app --host 0.0.0.0 --port 8000

import os
import tempfile
import json
from pathlib import Path
from typing import Optional

try:
    from fastapi import FastAPI, File, UploadFile, HTTPException
    from fastapi.responses import JSONResponse
except ImportError:
    raise ImportError(
        "FastAPI is required to run this API. Please install it using 'pip install fastapi[all]'"
    )

try:
    import torch
except ImportError:
    raise ImportError(
        "PyTorch is required to run this API. Please install it using 'pip install torch'."
    )

# Import analysis functions and model class (needed for loading saved models)
from demographics_detector import load_model_and_labels, predict_one, MultiHeadResNet
from skin_tone_classifier import classify_skin_tone_from_image
from anomaly_detector import detect_anomalies

# Import the full pipeline from main.py
try:
    from main import run_full_pipeline
except ImportError:
    run_full_pipeline = None

app = FastAPI(
    title="Face Analysis API",
    description="API for demographics, skin tone, and anomaly detection",
    version="1.0.0"
)

# Global variables for loaded models (loaded once at startup)
DEMOGRAPHICS_MODEL = None
DEMOGRAPHICS_GENDER_LABELS = None
DEMOGRAPHICS_AGE_LABELS = None
DEMOGRAPHICS_RACE_LABELS = None
DEMOGRAPHICS_DEVICE = None


@app.on_event("startup")
async def load_models():
    """Load models at startup for better performance."""
    global DEMOGRAPHICS_MODEL, DEMOGRAPHICS_GENDER_LABELS
    global DEMOGRAPHICS_AGE_LABELS, DEMOGRAPHICS_RACE_LABELS, DEMOGRAPHICS_DEVICE
    
    try:
        model_path = Path("models/fairface/fairface_model.pt")
        label_path = Path("models/fairface/fairface_label_dict.json")
        
        if not model_path.exists():
            print(f"[WARNING] Demographics model not found at {model_path}. Endpoint will fail at runtime.")
            return
        
        if not label_path.exists():
            print(f"[WARNING] Demographics labels not found at {label_path}. Endpoint will fail at runtime.")
            return
        
        DEMOGRAPHICS_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[i] Loading demographics model on device: {DEMOGRAPHICS_DEVICE}")
        
        # Run synchronous model loading
        import asyncio
        loop = asyncio.get_event_loop()
        DEMOGRAPHICS_MODEL, DEMOGRAPHICS_GENDER_LABELS, DEMOGRAPHICS_AGE_LABELS, DEMOGRAPHICS_RACE_LABELS = await loop.run_in_executor(
            None,
            load_model_and_labels,
            model_path,
            label_path,
            DEMOGRAPHICS_DEVICE
        )
        print("[i] Demographics model loaded successfully")
    except Exception as e:
        import traceback
        print(f"[ERROR] Failed to load demographics model: {e}")
        print(traceback.format_exc())
        print("[WARNING] /analyze/demographics and /analyze/full endpoints will not work")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Face Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "/analyze/demographics": "Analyze demographics (age, gender, ethnicity)",
            "/analyze/skintone": "Classify skin tone",
            "/analyze/anomalies": "Detect anomalies using YOLOv8",
            "/analyze/full": "Combined analysis (all three)",
            "/analyze/pipeline": "Full Week 6 pipeline (face detection, embedding, recognition, demographics, skin tone, anomalies)"
        }
    }


def _ensure_demographics_model_loaded():
    """Ensure demographics model is loaded, load it if not already loaded."""
    global DEMOGRAPHICS_MODEL, DEMOGRAPHICS_GENDER_LABELS
    global DEMOGRAPHICS_AGE_LABELS, DEMOGRAPHICS_RACE_LABELS, DEMOGRAPHICS_DEVICE
    
    if DEMOGRAPHICS_MODEL is not None:
        return  # Already loaded
    
    try:
        model_path = Path("models/fairface/fairface_model.pt")
        label_path = Path("models/fairface/fairface_label_dict.json")
        
        if not model_path.exists():
            raise FileNotFoundError(f"Demographics model not found at {model_path}")
        
        if not label_path.exists():
            raise FileNotFoundError(f"Demographics labels not found at {label_path}")
        
        DEMOGRAPHICS_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[i] Loading demographics model on device: {DEMOGRAPHICS_DEVICE}")
        
        # Fix for models saved as full objects from __main__
        # Map __main__ to demographics_detector for unpickling
        import demographics_detector
        
        # Try to load with custom class mapping
        try:
            # First try the normal loading
            DEMOGRAPHICS_MODEL, DEMOGRAPHICS_GENDER_LABELS, DEMOGRAPHICS_AGE_LABELS, DEMOGRAPHICS_RACE_LABELS = load_model_and_labels(
                model_path, label_path, DEMOGRAPHICS_DEVICE
            )
        except (AttributeError, RuntimeError) as e:
            if "Can't get attribute" in str(e) or "MultiHeadResNet" in str(e):
                # Model was saved as full object from __main__, need to fix it
                print("[i] Model was saved as full object, attempting to extract state_dict...")
                # Map __main__ to demographics_detector for unpickling
                import sys
                old_main = sys.modules.get('__main__')
                sys.modules['__main__'] = demographics_detector
                try:
                    # Now try loading
                    loaded = torch.load(model_path, map_location=DEMOGRAPHICS_DEVICE, weights_only=False)
                    # If it's a full model, extract state_dict
                    if hasattr(loaded, 'state_dict'):
                        # It's a model object, we need to rebuild and load state_dict
                        with open(label_path, "r") as f:
                            labels = json.load(f)
                        model = MultiHeadResNet(
                            n_gender=len(labels["gender_labels"]),
                            n_age=len(labels["age_labels"]),
                            n_race=len(labels["race_labels"]),
                            pretrained=False
                        ).to(DEMOGRAPHICS_DEVICE)
                        model.load_state_dict(loaded.state_dict())
                        model.eval()
                        DEMOGRAPHICS_MODEL = model
                        DEMOGRAPHICS_GENDER_LABELS = labels["gender_labels"]
                        DEMOGRAPHICS_AGE_LABELS = labels["age_labels"]
                        DEMOGRAPHICS_RACE_LABELS = labels["race_labels"]
                    else:
                        # Should have worked with normal loading
                        DEMOGRAPHICS_MODEL, DEMOGRAPHICS_GENDER_LABELS, DEMOGRAPHICS_AGE_LABELS, DEMOGRAPHICS_RACE_LABELS = load_model_and_labels(
                            model_path, label_path, DEMOGRAPHICS_DEVICE
                        )
                finally:
                    if old_main:
                        sys.modules['__main__'] = old_main
            else:
                raise
        print("[i] Demographics model loaded successfully")
    except Exception as e:
        import traceback
        error_msg = str(e)
        print(f"[ERROR] Failed to load demographics model: {error_msg}")
        print(traceback.format_exc())
        # Check if it's a pickle error - if so, the model might be saved incorrectly
        if "Can't get attribute" in error_msg or "MultiHeadResNet" in error_msg:
            raise RuntimeError("Model file appears to be saved incorrectly. Please retrain the model using train_demographics.py to save state_dict instead of full model object.")
        raise

@app.post("/analyze/demographics")
async def analyze_demographics(file: UploadFile = File(...)):
    """
    Analyze demographics (age, gender, ethnicity) from a face image.
    
    - **file**: Image file (JPEG, PNG, etc.)
    
    Returns JSON with demographics prediction.
    """
    # Try to load model if not already loaded
    try:
        _ensure_demographics_model_loaded()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Demographics model not available: {str(e)}")
    
    tmp_path = None
    # Save uploaded file temporarily
    try:
        suffix = Path(file.filename).suffix if file.filename else ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Run demographics analysis - convert Path to string
        result = predict_one(
            str(tmp_path),
            DEMOGRAPHICS_MODEL,
            DEMOGRAPHICS_GENDER_LABELS,
            DEMOGRAPHICS_AGE_LABELS,
            DEMOGRAPHICS_RACE_LABELS,
            DEMOGRAPHICS_DEVICE
        )
        
        return JSONResponse(content={
            "status": "success",
            "analysis": "demographics",
            "result": result
        })
    except Exception as e:
        import traceback
        error_detail = f"Analysis failed: {str(e)}"
        print(f"[ERROR] Demographics analysis error: {error_detail}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_detail)
    finally:
        # Clean up temporary file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass


@app.post("/analyze/skintone")
async def analyze_skintone(file: UploadFile = File(...)):
    """
    Classify skin tone from a face image.
    
    - **file**: Image file (JPEG, PNG, etc.)
    
    Returns JSON with skin tone classification.
    """
    tmp_path = None
    # Save uploaded file temporarily
    try:
        suffix = Path(file.filename).suffix if file.filename else ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Run skin tone analysis - convert Path to string
        result = classify_skin_tone_from_image(str(tmp_path))
        
        return JSONResponse(content={
            "status": "success",
            "analysis": "skintone",
            "result": result
        })
    except Exception as e:
        import traceback
        error_detail = f"Analysis failed: {str(e)}"
        print(f"[ERROR] Skin tone analysis error: {error_detail}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_detail)
    finally:
        # Clean up temporary file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass


@app.post("/analyze/anomalies")
async def analyze_anomalies(
    file: UploadFile = File(...),
    confidence: Optional[float] = 0.25
):
    """
    Detect acne and pigmentation anomalies in an image.
    
    - **file**: Image file (JPEG, PNG, etc.)
    - **confidence**: Confidence threshold (0.0 to 1.0, default: 0.25)
    
    Returns JSON with acne and pigmentation detection results.
    """
    if confidence is None or not 0.0 <= confidence <= 1.0:
        raise HTTPException(status_code=400, detail="Confidence must be between 0.0 and 1.0")
    
    tmp_path = None
    # Save uploaded file temporarily
    try:
        suffix = Path(file.filename).suffix if file.filename else ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as tmp_output_dir:
            # Run anomaly detection - convert Path to string
            result = detect_anomalies(
                str(tmp_path),
                model_path=None,  # type: ignore  # Use default model
                confidence=confidence or 0.25,
                output_dir=tmp_output_dir
            )
            
            # Remove file paths from result (they're temporary)
            result.pop("output_image", None)
            result.pop("json_output", None)
            
            return JSONResponse(content={
                "status": "success",
                "analysis": "anomalies",
                "result": result
            })
    except Exception as e:
        import traceback
        error_detail = f"Analysis failed: {str(e)}"
        print(f"[ERROR] Anomaly detection error: {error_detail}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_detail)
    finally:
        # Clean up temporary file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass


@app.post("/analyze/full")
async def analyze_full(
    file: UploadFile = File(...),
    confidence: Optional[float] = 0.25
):
    """
    Combined analysis: demographics, skin tone, and anomalies.
    
    - **file**: Image file (JPEG, PNG, etc.)
    - **confidence**: Confidence threshold for anomaly detection (0.0 to 1.0, default: 0.25)
    
    Returns JSON with all three analyses.
    """
    if confidence is None or not 0.0 <= confidence <= 1.0:
        raise HTTPException(status_code=400, detail="Confidence must be between 0.0 and 1.0")
    
    tmp_path = None
    # Save uploaded file temporarily
    try:
        suffix = Path(file.filename).suffix if file.filename else ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        results = {}
        errors = {}
        
            # 1. Demographics analysis
        try:
            _ensure_demographics_model_loaded()
            demographics_result = predict_one(
                str(tmp_path),
                DEMOGRAPHICS_MODEL,
                DEMOGRAPHICS_GENDER_LABELS,
                DEMOGRAPHICS_AGE_LABELS,
                DEMOGRAPHICS_RACE_LABELS,
                DEMOGRAPHICS_DEVICE
            )
            results["demographics"] = demographics_result
        except Exception as e:
            errors["demographics"] = str(e)
        
        # 2. Skin tone analysis
        try:
            skintone_result = classify_skin_tone_from_image(str(tmp_path))
            results["skintone"] = skintone_result
        except Exception as e:
            errors["skintone"] = str(e)
        
        # 3. Anomaly detection
        try:
            with tempfile.TemporaryDirectory() as tmp_output_dir:
                anomaly_result = detect_anomalies(
                    str(tmp_path),
                    model_path=None,  # type: ignore  # Use default model
                    confidence=confidence or 0.25,
                    output_dir=tmp_output_dir
                )
                anomaly_result.pop("output_image", None)
                anomaly_result.pop("json_output", None)
                results["anomalies"] = anomaly_result
        except Exception as e:
            errors["anomalies"] = str(e)
        
        return JSONResponse(content={
            "status": "success" if results else "partial_failure",
            "analysis": "full",
            "results": results,
            "errors": errors if errors else None
        })
    except Exception as e:
        import traceback
        error_detail = f"Analysis failed: {str(e)}"
        print(f"[ERROR] Full analysis error: {error_detail}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_detail)
    finally:
        # Clean up temporary file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass


@app.post("/analyze/pipeline")
async def analyze_pipeline(
    file: UploadFile = File(...),
    confidence: Optional[float] = 0.25
):
    """
    Full Week 6 pipeline analysis: Face detection, embedding generation, 
    face recognition, demographics, skin tone, and anomaly detection.
    
    - **file**: Image file (JPEG, PNG, etc.)
    - **confidence**: Confidence threshold for anomaly detection (0.0 to 1.0, default: 0.25)
    
    Returns JSON with complete pipeline analysis results.
    """
    if run_full_pipeline is None:
        raise HTTPException(
            status_code=503, 
            detail="Pipeline function not available. Ensure main.py is importable."
        )
    
    if not (0.0 <= (confidence or 0.0) <= 1.0):  # type: ignore
        raise HTTPException(status_code=400, detail="Confidence must be between 0.0 and 1.0")
    
    tmp_path = None
    try:
        # Save uploaded file temporarily
        suffix = Path(file.filename).suffix if file.filename else ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Use temporary directory for pipeline outputs
        with tempfile.TemporaryDirectory() as tmp_output_dir:
            # Run the full pipeline
            results = run_full_pipeline(str(tmp_path), tmp_output_dir)
            
            return JSONResponse(content={
                "status": "success",
                "analysis": "full_pipeline",
                "results": results
            })
    except Exception as e:
        import traceback
        error_detail = f"Pipeline analysis failed: {str(e)}"
        print(f"[ERROR] Pipeline analysis error: {error_detail}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_detail)
    finally:
        # Clean up temporary file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass


if __name__ == "__main__":
    try:
        import uvicorn
    except ImportError:
        raise ImportError(
            "uvicorn is required to run this server. "
            "Install it with 'pip install uvicorn'."
        )
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)

