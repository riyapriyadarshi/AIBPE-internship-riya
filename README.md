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
