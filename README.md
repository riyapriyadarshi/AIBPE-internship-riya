AIBPE-internship/
│
├── facedetector.py          # Face detection script
├── README.md                # Project documentation
├── sample.jpg               # Test input image
│
├── outputs/                 # Generated detections and cropped faces
│     ├── detections.jpg
│     ├── face_1.jpg
│     └── face_2.jpg
│
└── aibpeenv/                # Virtual environment (auto-generated)

-Create and activate virtual 
python -m venv aibpeenv
.\aibpeenv\Scripts\Activate.ps1

-Install required libraries
pip install --upgrade pip
pip install opencv-python numpy imutils

-How to run
python facedetector.py --input sample.jpg --out outputs --save-crops


| Argument         | Description                               |
| ---------------- | ----------------------------------------- |
| `--input` / `-i` | Path to input image                       |
| `--out` / `-o`   | Output directory                          |
| `--save-crops`   | Save cropped face regions (optional flag) |

