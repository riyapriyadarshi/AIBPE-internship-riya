# main.py
# Week 6: Final Integration - AIBPE Prototype
# 
# End-to-end pipeline integrating all modules:
# 1. Face detection
# 2. Embedding generation
# 3. Face recognition
# 4. Demographics detection
# 5. Skin tone classification
# 6. Anomaly detection
#
# Usage: python main.py [--image path/to/image.jpg]

import os
import json
import argparse
from pathlib import Path
import tempfile
import sys

# Import modules
from facedetector import load_cascade, process_image
from embedding_model import FaceEmbeddingModel
from recognize_face import recognize, load_db
from demographics_detector import load_model_and_labels, predict_one
from skin_tone_classifier import classify_skin_tone_from_image
from anomaly_detector import detect_anomalies

import torch
import cv2


def detect_faces_in_image(image_path, output_dir="outputs", save_crops=False):
    """
    Detect faces in an image and return metadata.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save outputs
        save_crops: Whether to save cropped face images
        
    Returns:
        Dictionary with face detection results
    """
    cascade = load_cascade()
    result = process_image(image_path, output_dir, cascade, save_crops)
    return result


def generate_embedding_for_image(image_path):
    """
    Generate embedding for an image.
    
    Args:
        image_path: Path to input image
        
    Returns:
        Numpy array with L2-normalized embedding
    """
    model = FaceEmbeddingModel()
    embedding = model.extract(image_path)
    return embedding.tolist()  # Convert to list for JSON serialization


def recognize_face_in_image(image_path, db_path="models/face_embeddings.json", threshold=0.55):
    """
    Recognize face in image by comparing with database.
    
    Args:
        image_path: Path to query image
        db_path: Path to embeddings database
        threshold: Similarity threshold
        
    Returns:
        Dictionary with recognition results
    """
    if not Path(db_path).exists():
        return {
            "identity": "unknown",
            "score": 0.0,
            "match_file": None,
            "error": "Embeddings database not found"
        }
    
    try:
        result = recognize(image_path, db_path, threshold)
        return result
    except Exception as e:
        return {
            "identity": "unknown",
            "score": 0.0,
            "match_file": None,
            "error": str(e)
        }


def detect_demographics(image_path, model_path="models/fairface/fairface_model.pt", 
                       label_path="models/fairface/fairface_label_dict.json"):
    """
    Detect demographics (age, gender, ethnicity) from an image.
    
    Args:
        image_path: Path to input image
        model_path: Path to demographics model
        label_path: Path to label dictionary
        
    Returns:
        Dictionary with demographics results
    """
    if not Path(model_path).exists() or not Path(label_path).exists():
        return {
            "error": "Demographics model not found",
            "gender": None,
            "age": None,
            "ethnicity": None
        }
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Handle models saved from __main__ by temporarily mapping __main__ to demographics_detector
        import demographics_detector
        
        # Store original __main__ module
        original_main = sys.modules.get('__main__')
        
        try:
            # Try loading normally first
            model, gender_labels, age_labels, race_labels = load_model_and_labels(
                model_path, label_path, device
            )
        except (AttributeError, RuntimeError) as e:
            if "Can't get attribute" in str(e) and "MultiHeadResNet" in str(e):
                # Model was saved as full object from __main__, need module mapping
                sys.modules['__main__'] = demographics_detector
                try:
                    model, gender_labels, age_labels, race_labels = load_model_and_labels(
                        model_path, label_path, device
                    )
                finally:
                    # Restore original __main__
                    if original_main:
                        sys.modules['__main__'] = original_main
            else:
                raise
        
        result = predict_one(image_path, model, gender_labels, age_labels, race_labels, device)
        return result
    except Exception as e:
        return {
            "error": str(e),
            "gender": None,
            "age": None,
            "ethnicity": None
        }


def classify_skin_tone(image_path):
    """
    Classify skin tone from an image.
    
    Args:
        image_path: Path to input image
        
    Returns:
        Dictionary with skin tone classification results
    """
    try:
        result = classify_skin_tone_from_image(image_path)
        return result
    except Exception as e:
        return {
            "error": str(e),
            "skin_tone_category": None,
            "lab_values": None
        }


def detect_anomalies_in_image(image_path, confidence=0.25):
    """
    Detect anomalies in an image using YOLOv8.
    
    Args:
        image_path: Path to input image
        confidence: Confidence threshold
        
    Returns:
        Dictionary with anomaly detection results
    """
    try:
        # Use temporary directory for outputs to avoid cluttering
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = detect_anomalies(
                image_path,
                model_path=None,  # type: ignore  # Use default model
                confidence=confidence,
                output_dir=tmp_dir
            )
            # Remove file paths from result (they're temporary)
            result.pop("output_image", None)
            result.pop("json_output", None)
            return result
    except Exception as e:
        return {
            "error": str(e),
            "detections": [],
            "total_detections": 0
        }


def run_full_pipeline(image_path, output_dir="outputs"):
    """
    Run the complete AIBPE pipeline on an image.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save outputs
        
    Returns:
        Dictionary with all analysis results
    """
    print(f"\n{'='*60}")
    print(f"AIBPE Pipeline - Processing: {image_path}")
    print(f"{'='*60}\n")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize results dictionary
    results = {
        "input_image": str(image_path),
        "face_detection": {},
        "embedding": None,
        "face_recognition": {},
        "demographics": {},
        "skin_tone": {},
        "anomaly_detection": {}
    }
    
    # Step 1: Face Detection
    print("[1/6] Detecting faces...")
    try:
        face_detection = detect_faces_in_image(image_path, output_dir, save_crops=True)
        results["face_detection"] = face_detection or {"error": "Failed to detect faces", "detections": []}
        print(f"    Found {len(results['face_detection'].get('detections', []))} face(s)")
    except Exception as e:
        print(f"    ERROR: {e}")
        results["face_detection"] = {"error": str(e), "detections": []}
    
    # Step 2: Generate Embedding
    print("[2/6] Generating embedding...")
    try:
        embedding = generate_embedding_for_image(image_path)
        results["embedding"] = embedding
        print(f"    Generated {len(embedding)}-dimensional embedding")
    except Exception as e:
        print(f"    ERROR: {e}")
        results["embedding"] = None
    
    # Step 3: Face Recognition
    print("[3/6] Recognizing face...")
    try:
        recognition = recognize_face_in_image(image_path)
        results["face_recognition"] = recognition
        if recognition.get("identity") != "unknown":
            print(f"    Identity: {recognition.get('identity')} (score: {recognition.get('score', 0):.3f})")
        else:
            print(f"    Identity: unknown (score: {recognition.get('score', 0):.3f})")
    except Exception as e:
        print(f"    ERROR: {e}")
        results["face_recognition"] = {"error": str(e)}
    
    # Step 4: Demographics Detection
    print("[4/6] Detecting demographics...")
    try:
        demographics = detect_demographics(image_path)
        results["demographics"] = demographics
        if "error" not in demographics:
            print(f"    Gender: {demographics.get('gender')}, Age: {demographics.get('age')}, Ethnicity: {demographics.get('ethnicity')}")
        else:
            print(f"    ERROR: {demographics.get('error')}")
    except Exception as e:
        print(f"    ERROR: {e}")
        results["demographics"] = {"error": str(e)}
    
    # Step 5: Skin Tone Classification
    print("[5/6] Classifying skin tone...")
    try:
        skin_tone = classify_skin_tone(image_path)
        results["skin_tone"] = skin_tone
        if "error" not in skin_tone:
            print(f"    Category: {skin_tone.get('skin_tone_category')}")
        else:
            print(f"    ERROR: {skin_tone.get('error')}")
    except Exception as e:
        print(f"    ERROR: {e}")
        results["skin_tone"] = {"error": str(e)}
    
    # Step 6: Anomaly Detection
    print("[6/6] Detecting anomalies...")
    try:
        anomalies = detect_anomalies_in_image(image_path)
        results["anomaly_detection"] = anomalies
        if "error" not in anomalies:
            print(f"    Found {anomalies.get('total_detections', 0)} detection(s)")
        else:
            print(f"    ERROR: {anomalies.get('error')}")
    except Exception as e:
        print(f"    ERROR: {e}")
        results["anomaly_detection"] = {"error": str(e)}
    
    # Save results to JSON
    output_filename = Path(image_path).stem
    output_json_path = os.path.join(output_dir, f"{output_filename}_full_analysis.json")
    
    print(f"\n{'='*60}")
    print(f"Saving results to: {output_json_path}")
    print(f"{'='*60}\n")
    
    with open(output_json_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print("Pipeline completed successfully!")
    print(f"Results saved to: {output_json_path}")
    
    # Return results for API usage
    return results


def main():
    parser = argparse.ArgumentParser(
        description="AIBPE Prototype - Complete Face Analysis Pipeline"
    )
    parser.add_argument(
        "--image", "-i",
        default=None,
        help="Path to input image (default: sample.jpg or sample.jpeg)"
    )
    parser.add_argument(
        "--output", "-o",
        default="outputs",
        help="Output directory (default: outputs)"
    )
    
    args = parser.parse_args()
    
    # Determine input image
    if args.image:
        image_path = Path(args.image)
    else:
        # Try to find sample image
        sample_jpg = Path("sample.jpg")
        sample_jpeg = Path("sample.jpeg")
        
        if sample_jpg.exists():
            image_path = sample_jpg
        elif sample_jpeg.exists():
            image_path = sample_jpeg
        else:
            print("ERROR: No input image provided and no sample.jpg or sample.jpeg found.")
            print("Please provide an image using --image flag.")
            return
    
    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}")
        return
    
    # Run pipeline
    try:
        results = run_full_pipeline(str(image_path), args.output)
        
        # Print summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Input: {results['input_image']}")
        print(f"Faces detected: {len(results['face_detection'].get('detections', []))}")
        print(f"Embedding generated: {'Yes' if results['embedding'] else 'No'}")
        print(f"Face recognized: {results['face_recognition'].get('identity', 'N/A')}")
        if 'error' not in results['demographics']:
            print(f"Demographics: {results['demographics'].get('gender')}, {results['demographics'].get('age')}, {results['demographics'].get('ethnicity')}")
        if 'error' not in results['skin_tone']:
            print(f"Skin tone: {results['skin_tone'].get('skin_tone_category')}")
        print(f"Anomalies detected: {results['anomaly_detection'].get('total_detections', 0)}")
        print("="*60)
        
    except Exception as e:
        print(f"\nERROR: Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

