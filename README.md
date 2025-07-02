<p align="left">
  <img src="mmocr3.png" alt="mmocr_anonymizer banner" width="300"/>
</p>

<b>DICOM Text Masker – Automatic In-Burned Text Removal</b>
Description:
This project provides a lightweight Python tool for automatically detecting and masking in-burned patient information in DICOM ultrasound images. It uses MMOCR (DBNet) for text detection and applies masking directly to the pixel data before saving a sanitized DICOM file.

<b>Features:</b>

Loads .dcm files using pydicom
Detects in-image text with MMOCR (no OCR required)
Masks detected regions with black rectangles
Saves anonymized DICOM files for secure sharing

<b>Use Cases:</b>

Medical image anonymization
Pre-processing for dataset sharing
Automated privacy protection in clinical workflows

<b>Requirements:</b>

Python 3.8+
PyTorch, MMOCR, OpenCV, pydicom

# mmocr_anonymizer
