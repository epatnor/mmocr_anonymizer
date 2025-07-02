import streamlit as st
import numpy as np
import cv2
import pydicom
import tempfile
from mmocr.apis import MMOCR
from io import BytesIO

st.set_page_config(page_title="mmocr_anonymizer", layout="centered")
st.title("🩻 mmocr_anonymizer")
st.caption("Automatic in-image anonymization of DICOM ultrasound files")

@st.cache_resource
def load_ocr():
    return MMOCR(det='DBNet', recog=None)

def dicom_to_image(dicom_bytes):
    ds = pydicom.dcmread(BytesIO(dicom_bytes))
    img = ds.pixel_array.astype(np.uint8)

    if ds.PhotometricInterpretation == "MONOCHROME1":
        img = np.max(img) - img

    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return ds, img

def mask_text(image, boxes):
    masked = image.copy()
    for box in boxes:
        pts = np.array(box, dtype=np.int32)
        cv2.fillPoly(masked, [pts], (0))
    return masked

def save_masked_dicom(ds, masked_img):
    ds.PixelData = masked_img.astype(ds.pixel_array.dtype).tobytes()
    with BytesIO() as buffer:
        ds.save_as(buffer)
        return buffer.getvalue()

# --- GUI ---

uploaded_file = st.file_uploader("📤 Upload a DICOM file", type=["dcm"])

if uploaded_file:
    dicom_bytes = uploaded_file.read()
    ds, img = dicom_to_image(dicom_bytes)
    st.image(img, caption="Original Image", use_column_width=True, clamp=True)

    if st.button("🔍 Detect and Mask Text"):
        st.info("Running MMOCR...")
        ocr = load_ocr()
        result = ocr.readtext(img)

        if result and "boundary_result" in result[0]:
            boxes = result[0]["boundary_result"]
            st.success(f"Detected {len(boxes)} regions")
            masked_img = mask_text(img, boxes)
            st.image(masked_img, caption="Masked Image", use_column_width=True, clamp=True)

            masked_dicom_bytes = save_masked_dicom(ds, masked_img)
            st.download_button(
                label="💾 Download Masked DICOM",
                data=masked_dicom_bytes,
                file_name="masked.dcm",
                mime="application/dicom"
            )
        else:
            st.warning("No text regions detected.")

