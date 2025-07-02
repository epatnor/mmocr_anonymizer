🩺 DICOM Text Masker – Automatic In-Burned Text Removal
Description:
This project provides a lightweight Python tool for automatically detecting and masking in-burned patient information in DICOM ultrasound images. It uses MMOCR (DBNet) for text detection and applies masking directly to the pixel data before saving a sanitized DICOM file.

Features:

Loads .dcm files using pydicom
Detects in-image text with MMOCR (no OCR required)
Masks detected regions with black rectangles
Saves anonymized DICOM files for secure sharing

Use Cases:

Medical image anonymization
Pre-processing for dataset sharing
Automated privacy protection in clinical workflows

Requirements:

Python 3.8+
PyTorch, MMOCR, OpenCV, pydicom

# mmocr_anonymizer
