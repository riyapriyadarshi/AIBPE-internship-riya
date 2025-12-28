# anomaly_detector.py
# Week 4: Anomaly Detection using YOLOv8
#
# Tasks:
# 1. Download YOLOv8 nano pretrained model
# 2. Perform inference for acne/pigmentation detection
# 3. Build bounding-box drawing utility for detected anomalies
# 4. Output: anomaly_detector.py
#
# Usage:
#   python anomaly_detector.py --input image.jpg
#   python anomaly_detector.py --input image.jpg --output results/
#   python anomaly_detector.py --input image.jpg --confidence 0.25

import cv2
import numpy as np
import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import os

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] ultralytics package not installed.")
    print("Please install it using: pip install ultralytics")
    exit(1)

# YOLOv8 model configuration
YOLO_MODEL_NAME = "yolov8n.pt"  # YOLOv8 nano (smallest, fastest)
MODEL_DIR = "models/yolo"

def download_yolo_model(model_name: str = YOLO_MODEL_NAME) -> str:
    """
    Download YOLOv8 nano pretrained model if not already present.
    
    Args:
        model_name: Name of the model file (e.g., 'yolov8n.pt')
        
    Returns:
        Path to the model file
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, model_name)
    
    if os.path.exists(model_path):
        print(f"[i] Model already exists: {model_path}")
        return model_path
    
    print(f"[i] Downloading YOLOv8 nano model: {model_name}")
    print("[i] This may take a few moments...")
    
    try:
        # YOLO will automatically download the model on first use
        model = YOLO(model_name)
        # Save to our models directory
        import shutil
        if os.path.exists(model_name):
            shutil.move(model_name, model_path)
        return model_path
    except Exception as e:
        print(f"[ERROR] Failed to download model: {e}")
        # Try loading directly (ultralytics handles download)
        model = YOLO(model_name)
        return model_name  # ultralytics may store it in current dir

def load_yolo_model(model_path: str = None): # type: ignore
    """
    Load YOLOv8 model.
    
    Args:
        model_path: Path to model file (if None, uses default)
        
    Returns:
        YOLO model object
    """
    if model_path is None:
        model_path = download_yolo_model()
    
    print(f"[i] Loading YOLOv8 model from: {model_path}")
    try:
        model = YOLO(model_path)
        print("[i] Model loaded successfully")
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        raise

def detect_and_crop_face(image_path: str) -> Optional[np.ndarray]:
    """
    Detect and crop face region from image for skin analysis.
    
    Args:
        image_path: Path to input image
        
    Returns:
        Cropped face image (BGR) or None if no face detected
    """
    # Use Haar Cascade for face detection
    try:
        from facedetector import load_cascade, detect_faces, crop_face
    except ImportError:
        print("[WARNING] facedetector module not found, using OpenCV Haar Cascade directly")
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return None
        
        # Use the largest face
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]
        
        # Expand crop slightly for better analysis
        margin = int(min(w, h) * 0.1)
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(img.shape[1] - x, w + 2 * margin)
        h = min(img.shape[0] - y, h + 2 * margin)
        
        return img[y:y+h, x:x+w]
    
    # Use facedetector module if available
    cascade = load_cascade()
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(cascade, img)
    
    if len(faces) == 0:
        return None
    
    # Use the largest face
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces[0]
    
    # Expand crop slightly for better analysis
    margin = int(min(w, h) * 0.1)
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(img.shape[1] - x, w + 2 * margin)
    h = min(img.shape[0] - y, h + 2 * margin)
    
    return crop_face(img, x, y, w, h)

def detect_acne_and_pigmentation(face_image: np.ndarray, 
                                 acne_threshold: float = 0.15,
                                 pigmentation_threshold: float = 0.12) -> List[Dict]:
    """
    Detect acne and pigmentation spots on face using image processing.
    
    Args:
        face_image: Cropped face image (BGR format)
        acne_threshold: Threshold for detecting red spots (acne) - lower = more sensitive
        pigmentation_threshold: Threshold for detecting dark spots - lower = more sensitive
        
    Returns:
        List of detection dictionaries with bounding boxes and anomaly types
    """
    detections = []
    face_h, face_w = face_image.shape[:2]
    face_area = face_h * face_w
    
    # Convert to different color spaces for better detection
    hsv = cv2.cvtColor(face_image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(face_image, cv2.COLOR_BGR2LAB)
    
    # Extract channels
    h, s, v = cv2.split(hsv)
    l, a, b = cv2.split(lab)
    b_chan, g, r = cv2.split(face_image)
    
    # Detect ACNE (red/inflamed spots)
    # Acne appears as localized red areas with high saturation
    # Method: Look for areas where R channel is significantly higher than G and B
    
    # Calculate red dominance (R should be > G and > B)
    r_float = r.astype(float)
    g_float = g.astype(float)
    b_float = b_chan.astype(float)
    
    # Red areas: R > 1.15*G and R > 1.15*B (more restrictive to find actual red spots)
    red_condition = (r_float > (g_float * 1.15)) & (r_float > (b_float * 1.15))
    saturation_condition = s > 60  # Higher saturation threshold for acne
    brightness_condition = (v > 40) & (v < 220)  # Not too dark or too bright
    
    acne_mask = red_condition & saturation_condition & brightness_condition
    acne_mask = acne_mask.astype(np.uint8) * 255
    
    # Morphological operations to find localized spots
    # Remove noise with opening
    kernel_small = np.ones((3, 3), np.uint8)
    acne_mask = cv2.morphologyEx(acne_mask, cv2.MORPH_OPEN, kernel_small)
    # Close small gaps
    kernel_close = np.ones((5, 5), np.uint8)
    acne_mask = cv2.morphologyEx(acne_mask, cv2.MORPH_CLOSE, kernel_close)
    # Smooth
    acne_mask = cv2.medianBlur(acne_mask, 5)
    
    # Find contours for acne
    acne_contours, _ = cv2.findContours(acne_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter acne detections - should be small localized spots
    max_acne_area = face_area * 0.05  # Max 5% of face (to avoid detecting whole face)
    min_acne_area = face_area * 0.0003  # Min 0.03% of face (small spots)
    
    for contour in acne_contours:
        area = cv2.contourArea(contour)
        if min_acne_area < area < max_acne_area:  # Size filter
            x, y, w, h = cv2.boundingRect(contour)
            # Aspect ratio filter - spots should be roughly circular/oval
            aspect_ratio = max(w, h) / (min(w, h) + 1e-5)
            if aspect_ratio < 3.0:  # Not too elongated
                roi = face_image[max(0, y):min(face_h, y+h), max(0, x):min(face_w, x+w)]
                if roi.size > 0:
                    # Calculate redness intensity in ROI
                    roi_r = np.mean(roi[:, :, 2])
                    roi_g = np.mean(roi[:, :, 1])
                    roi_b = np.mean(roi[:, :, 0])
                    redness_ratio = roi_r / (roi_g + roi_b + 1e-5)
                    confidence = min(0.95, max(0.4, min(1.0, redness_ratio / 1.3)))
                    
                    detections.append({
                        "class": "acne",
                        "class_id": 0,
                        "confidence": float(confidence),
                        "bbox": {
                            "x1": float(x),
                            "y1": float(y),
                            "x2": float(x + w),
                            "y2": float(y + h),
                            "width": float(w),
                            "height": float(h)
                        },
                        "area": float(area)
                    })
    
    # Detect PIGMENTATION (dark spots)
    # Pigmentation appears as dark brown/black localized areas
    # Method: Use L channel from LAB - dark spots have low L values compared to surrounding skin
    
    l_normalized = l.astype(float) / 255.0
    
    # Calculate adaptive threshold based on local mean
    # Use a more sophisticated approach: find spots darker than surrounding area
    mean_lightness = np.mean(l_normalized)
    std_lightness = np.std(l_normalized)
    
    # Threshold: areas darker than mean - 1.5*std (but not too dark to avoid shadows)
    dark_threshold = max(0.25, mean_lightness - (std_lightness * 1.5))
    
    pigmentation_mask = l_normalized < dark_threshold
    pigmentation_mask = pigmentation_mask.astype(np.uint8) * 255
    
    # Morphological operations
    kernel_open = np.ones((3, 3), np.uint8)
    pigmentation_mask = cv2.morphologyEx(pigmentation_mask, cv2.MORPH_OPEN, kernel_open)
    kernel_close = np.ones((5, 5), np.uint8)
    pigmentation_mask = cv2.morphologyEx(pigmentation_mask, cv2.MORPH_CLOSE, kernel_close)
    pigmentation_mask = cv2.medianBlur(pigmentation_mask, 5)
    
    # Find contours for pigmentation
    pig_contours, _ = cv2.findContours(pigmentation_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter pigmentation detections - should be small localized spots
    max_pig_area = face_area * 0.08  # Max 8% of face
    min_pig_area = face_area * 0.0002  # Min 0.02% of face (small spots)
    
    for contour in pig_contours:
        area = cv2.contourArea(contour)
        if min_pig_area < area < max_pig_area:  # Size filter
            x, y, w, h = cv2.boundingRect(contour)
            # Aspect ratio filter
            aspect_ratio = max(w, h) / (min(w, h) + 1e-5)
            if aspect_ratio < 3.5:  # Not too elongated
                roi = face_image[max(0, y):min(face_h, y+h), max(0, x):min(face_w, x+w)]
                if roi.size > 0:
                    # Calculate relative darkness
                    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    roi_mean = np.mean(roi_gray)
                    relative_darkness = 1.0 - (roi_mean / 255.0)
                    
                    # Compare to surrounding area
                    expanded_y1 = max(0, y - h//2)
                    expanded_y2 = min(face_h, y + h + h//2)
                    expanded_x1 = max(0, x - w//2)
                    expanded_x2 = min(face_w, x + w + w//2)
                    surrounding = face_image[expanded_y1:expanded_y2, expanded_x1:expanded_x2]
                    if surrounding.size > 0:
                        surrounding_gray = cv2.cvtColor(surrounding, cv2.COLOR_BGR2GRAY)
                        surrounding_mean = np.mean(surrounding_gray)
                        darkness_contrast = (surrounding_mean - roi_mean) / 255.0
                        confidence = min(0.95, max(0.4, darkness_contrast * 2.0))
                    else:
                        confidence = min(0.95, max(0.4, relative_darkness * 1.5))
                    
                    detections.append({
                        "class": "pigmentation",
                        "class_id": 1,
                        "confidence": float(confidence),
                        "bbox": {
                            "x1": float(x),
                            "y1": float(y),
                            "x2": float(x + w),
                            "y2": float(y + h),
                            "width": float(w),
                            "height": float(h)
                        },
                        "area": float(area)
                    })
    
    return detections

def perform_inference(image_path: str, confidence: float = 0.25) -> Tuple[List[Dict], Optional[np.ndarray]]:
    """
    Perform inference for acne/pigmentation detection.
    
    Args:
        image_path: Path to input image
        confidence: Confidence threshold (used for filtering low-confidence detections)
        
    Returns:
        Tuple of (list of detections, cropped face image or None)
    """
    print(f"[i] Performing inference on: {image_path}")
    
    # First, detect and crop face
    face_image = detect_and_crop_face(image_path)
    
    if face_image is None:
        print("[WARNING] No face detected in image. Skipping acne/pigmentation detection.")
        return [], None
    
    print(f"[i] Face detected: {face_image.shape[1]}x{face_image.shape[0]} pixels")
    
    # Detect acne and pigmentation on the face
    detections = detect_acne_and_pigmentation(face_image, 
                                             acne_threshold=0.15,
                                             pigmentation_threshold=0.12)
    
    # Filter by confidence
    filtered_detections = [d for d in detections if d["confidence"] >= confidence]
    
    acne_count = sum(1 for d in filtered_detections if d["class"] == "acne")
    pig_count = sum(1 for d in filtered_detections if d["class"] == "pigmentation")
    
    print(f"[i] Detected {acne_count} acne spot(s) and {pig_count} pigmentation spot(s)")
    
    return filtered_detections, face_image

def draw_bounding_boxes(image: np.ndarray, detections: List[Dict], 
                        show_labels: bool = True, show_confidence: bool = True) -> np.ndarray:
    """
    Draw bounding boxes on image for detected anomalies.
    
    Args:
        image: Input image (BGR format)
        detections: List of detection dictionaries
        show_labels: Whether to show class labels
        show_confidence: Whether to show confidence scores
        
    Returns:
        Image with bounding boxes drawn
    """
    output_image = image.copy()
    
    # Color map for different anomaly types
    colors = {
        "acne": (0, 0, 255),          # Red for acne
        "pigmentation": (128, 0, 128)  # Purple for pigmentation
    }
    
    for det in detections:
        bbox = det["bbox"]
        x1, y1 = int(bbox["x1"]), int(bbox["y1"])
        x2, y2 = int(bbox["x2"]), int(bbox["y2"])
        
        # Get color for this anomaly type
        anomaly_type = det["class"]
        color = colors.get(anomaly_type, (0, 255, 0))
        
        # Draw bounding box
        cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label text
        label_parts = []
        if show_labels:
            label_parts.append(anomaly_type.upper())
        if show_confidence:
            label_parts.append(f"{det['confidence']:.2f}")
        
        if label_parts:
            label = " ".join(label_parts)
            
            # Calculate text size and position
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # Draw label background
            cv2.rectangle(
                output_image,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                output_image,
                label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
    
    return output_image

def detect_anomalies(image_path: str, model_path: str = None,  # type: ignore
                     confidence: float = 0.25, output_dir: str = "outputs") -> Dict:
    """
    Main function to detect anomalies (acne and pigmentation) in an image.
    
    Args:
        image_path: Path to input image
        model_path: Path to YOLOv8 model (kept for API compatibility, not used for acne/pigmentation)
        confidence: Confidence threshold for filtering detections
        output_dir: Directory to save output images and results
        
    Returns:
        Dictionary with detection results
    """
    # Load original image
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Perform inference for acne/pigmentation detection
    detections, face_image = perform_inference(image_path, confidence)
    
    # Draw bounding boxes on the face image (if face was detected)
    if face_image is not None and len(detections) > 0:
        annotated_face = draw_bounding_boxes(face_image.copy(), detections)
        
        # Scale detections back to original image coordinates if needed
        # For now, we'll annotate the face crop
        annotated_image_for_save = annotated_face
    else:
        # No face detected or no detections
        annotated_image_for_save = original_image.copy()
        if face_image is not None:
            annotated_image_for_save = face_image.copy()
    
    # Prepare results
    acne_count = sum(1 for d in detections if d["class"] == "acne")
    pigmentation_count = sum(1 for d in detections if d["class"] == "pigmentation")
    
    result = {
        "input_image": str(image_path),
        "detections": detections,
        "total_detections": len(detections),
        "acne_count": acne_count,
        "pigmentation_count": pigmentation_count,
        "face_detected": face_image is not None,
        "confidence_threshold": confidence
    }
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save annotated image
    input_name = Path(image_path).stem
    output_image_path = os.path.join(output_dir, f"{input_name}_anomalies.jpg")
    cv2.imwrite(output_image_path, annotated_image_for_save)
    result["output_image"] = output_image_path
    
    # Save JSON results
    json_path = os.path.join(output_dir, f"{input_name}_anomalies.json")
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    result["json_output"] = json_path
    
    # Also save the face crop if available
    if face_image is not None:
        face_crop_path = os.path.join(output_dir, f"{input_name}_face_crop.jpg")
        cv2.imwrite(face_crop_path, face_image)
        result["face_crop_image"] = face_crop_path
    
    return result

def main():
    parser = argparse.ArgumentParser(
        description="Anomaly Detection using YOLOv8 for acne/pigmentation detection"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--output", "-o",
        default="outputs",
        help="Output directory for results"
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="Path to YOLOv8 model file (kept for compatibility, not used for acne/pigmentation detection)"
    )
    parser.add_argument(
        "--confidence", "-c",
        type=float,
        default=0.25,
        help="Confidence threshold (0.0 to 1.0, default: 0.25)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed output"
    )
    
    args = parser.parse_args()
    
    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Input image not found: {args.input}")
        return
    
    # Validate confidence
    if not 0.0 <= args.confidence <= 1.0:
        print(f"[ERROR] Confidence must be between 0.0 and 1.0")
        return
    
    try:
        # Detect anomalies (acne and pigmentation)
        result = detect_anomalies(
            str(input_path),
            args.model,
            args.confidence,
            args.output
        )
        
        # Print results
        if args.verbose:
            print("=" * 60)
            print("ANOMALY DETECTION RESULTS")
            print("=" * 60)
            print(f"Input Image: {result['input_image']}")
            print(f"Face Detected: {result['face_detected']}")
            print(f"Confidence Threshold: {result['confidence_threshold']}")
            print(f"Total Detections: {result['total_detections']}")
            print(f"Acne Spots: {result['acne_count']}")
            print(f"Pigmentation Spots: {result['pigmentation_count']}")
            print(f"\nDetections:")
            for i, det in enumerate(result['detections'], 1):
                bbox = det['bbox']
                print(f"  {i}. {det['class'].upper()} (conf: {det['confidence']:.2f}, area: {det.get('area', 0):.0f} px²)")
                print(f"     BBox: ({bbox['x1']:.0f}, {bbox['y1']:.0f}) to ({bbox['x2']:.0f}, {bbox['y2']:.0f})")
            print(f"\nOutput Image: {result['output_image']}")
            print(f"JSON Results: {result['json_output']}")
            if 'face_crop_image' in result:
                print(f"Face Crop: {result['face_crop_image']}")
            print("=" * 60)
        else:
            print(f"Face Detected: {result['face_detected']}")
            print(f"Detected {result['acne_count']} acne spot(s) and {result['pigmentation_count']} pigmentation spot(s)")
            print(f"Results saved to: {result['output_image']}")
            print(f"JSON saved to: {result['json_output']}")
        
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        if args.verbose:
            traceback.print_exc()

if __name__ == "__main__":
    main()
