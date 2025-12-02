import os
import argparse
import cv2
import json

def get_args():
    p = argparse.ArgumentParser(description="Haar cascade face detection")
    p.add_argument("--input", "-i", required=True, help="Path to input image")
    p.add_argument("--out", "-o", default="outputs", help="Output directory")
    p.add_argument("--save-crops", action="store_true", help="Save cropped faces")
    return p.parse_args()

def load_cascade():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(cascade_path)

def detect_faces(cascade, image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50,50))
    return faces

def crop_face(img, x, y, w, h):
    return img[y:y+h, x:x+w]

def process_image(path, out_dir, cascade, save_crops=False):
    img = cv2.imread(path)
    if img is None:
        print(f"[!] Could not open: {path}")
        return None

    faces = detect_faces(cascade, img)
    os.makedirs(out_dir, exist_ok=True)

    out_vis = img.copy()
    meta = {"input": os.path.basename(path), "detections": []}

    for i, (x, y, w, h) in enumerate(faces, start=1):
        cv2.rectangle(out_vis, (x, y), (x+w, y+h), (0,255,0), 2)

        entry = {"id": i, "bbox": [int(x), int(y), int(w), int(h)]}

        if save_crops:
            crop = crop_face(img, x, y, w, h)
            crop_name = f"face_{i}.jpg"
            crop_path = os.path.join(out_dir, crop_name)
            cv2.imwrite(crop_path, crop)
            entry["crop"] = crop_name

        meta["detections"].append(entry)

    vis_path = os.path.join(out_dir, "detections.jpg")
    cv2.imwrite(vis_path, out_vis)

    meta_path = os.path.join(out_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[i] Detect {len(faces)} face(s). Output saved to: {out_dir}")
    return meta

def main():
    args = get_args()
    cascade = load_cascade()
    process_image(args.input, args.out, cascade, args.save_crops)

if __name__ == "__main__":
    main()
