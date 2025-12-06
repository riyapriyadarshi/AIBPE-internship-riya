import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import json
import cv2
import argparse
import os
import sys

# Model paths
MODEL_PATH = "models/fairface/fairface_model.pt"
LABEL_PATH = "models/fairface/fairface_label_dict.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model architecture first
class MultiHeadResNet(nn.Module):
    def __init__(self, n_gender, n_age, n_race, pretrained=True):
        super().__init__()
        # Use weights parameter for newer torchvision versions
        if pretrained:
            base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            base = models.resnet18(weights=None)
        in_feat = base.fc.in_features
        base.fc = nn.Identity()
        self.base = base
        self.gender_head = nn.Linear(in_feat, n_gender)
        self.age_head = nn.Linear(in_feat, n_age)
        self.race_head = nn.Linear(in_feat, n_race)

    def forward(self, x):
        feat = self.base(x)
        g = self.gender_head(feat)
        a = self.age_head(feat)
        r = self.race_head(feat)
        return g, a, r

# Make the class available in __main__ namespace for unpickling
sys.modules['__main__'].MultiHeadResNet = MultiHeadResNet

# Load label dictionary first to get number of classes
with open(LABEL_PATH, "r") as f:
    label_dict = json.load(f)

# Get number of classes from label dict
n_gender = len(label_dict["gender_labels"])
n_age = len(label_dict["age_labels"])
n_race = len(label_dict["race_labels"])

# Create model architecture
model = MultiHeadResNet(n_gender, n_age, n_race, pretrained=False).to(device)

# Load model - try full model first (current format), then try state_dict
try:
    # Try loading full model (for existing models)
    loaded = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    if isinstance(loaded, dict) and 'state_dict' in loaded:
        # If it's a dict with 'state_dict' key
        model.load_state_dict(loaded['state_dict'])
        print("[i] Loaded model from state_dict (dict format)")
    elif isinstance(loaded, dict):
        # If it's a state_dict directly
        model.load_state_dict(loaded)
        print("[i] Loaded model from state_dict")
    else:
        # If it's a full model object
        model = loaded.to(device)
        print("[i] Loaded full model (legacy format)")
except Exception as e:
    print(f"[!] Error loading model: {e}")
    print("[!] Please ensure the model file exists and is valid")
    raise

model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def detect_faces_opencv(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    return faces


def analyze_all_faces(image_path, save_crops=True, crop_dir="outputs"):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"[!] Cannot open: {image_path}")
        return []
    faces = detect_faces_opencv(img)
    if len(faces) == 0:
        print("[i] No faces detected.")
        return []

    os.makedirs(crop_dir, exist_ok=True)
    results = []
    face_number = 1
    for (x, y, w, h) in faces:
        crop = img[y:y+h, x:x+w]
        crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        crop_tensor = transform(crop_pil).unsqueeze(0)
        with torch.no_grad():
            outputs = model(crop_tensor)
        gender_out, age_out, race_out = outputs
        gender = label_dict["gender_labels"][torch.argmax(gender_out).item()]
        age = label_dict["age_labels"][torch.argmax(age_out).item()]
        race = label_dict["race_labels"][torch.argmax(race_out).item()]
        result = {
            "face_number": face_number,
            "bbox": [int(x), int(y), int(w), int(h)],
            "gender": gender,
            "age": age,
            "ethnicity": race
        }
        results.append(result)
        if save_crops:
            crop_filename = os.path.join(crop_dir, f"face_{face_number}.jpg")
            cv2.imwrite(crop_filename, crop)
            result["crop_image"] = crop_filename
        face_number += 1
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="Input image file")
    parser.add_argument("--out", "-o", default="outputs", help="Output directory for face crops and JSON")
    parser.add_argument("--save-crops", action="store_true", help="Save crop images")
    args = parser.parse_args()

    face_results = analyze_all_faces(args.input, save_crops=args.save_crops, crop_dir=args.out)
    print(json.dumps(face_results, indent=2))
    if face_results:
        out_json = os.path.join(args.out, "face_demographics.json")
        with open(out_json, "w") as f:
            json.dump(face_results, f, indent=2)
        print("[i] Results saved to", out_json)

if __name__ == "__main__":
    main()
