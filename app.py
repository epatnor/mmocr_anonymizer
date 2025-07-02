import streamlit as st
import numpy as np
import cv2
import pydicom
from io import BytesIO
from mmocr.apis import MMOCR

st.set_page_config(page_title="mmocr_anonymizer", layout="wide")
st.title("🩻 mmocr_anonymizer")
st.caption("Automatic anonymization of burned-in text in DICOM ultrasound images")

# --- Sidebar: settings ---
st.sidebar.header("⚙️ Detection Settings")
ocr_psm = st.sidebar.selectbox("OCR Mode (psm)", options=[6, 7, 11], index=0)
show_boxes = st.sidebar.checkbox("Show bounding boxes", value=False)

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

def mask_text(image, boxes, draw_boxes=False):
    masked = image.copy()
    for box in boxes:
        pts = np.array(box, dtype=np.int32)
        if draw_boxes:
            cv2.polylines(masked, [pts], isClosed=True, color=(255,), thickness=2)
        else:
            cv2.fillPoly(masked, [pts], (0))
    return masked

def save_masked_dicom(ds, masked_img):
    ds.PixelData = masked_img.astype(ds.pixel_array.dtype).tobytes()
    with BytesIO() as buffer:
        ds.save_as(buffer)
        return buffer.getvalue()

# --- Columns layout ---
col1, col2 = st.columns(2, gap="medium")

with col1:
    st.subheader("📤 Original DICOM")
    uploaded_file = st.file_uploader("Upload .dcm file", type=["dcm"])
    if uploaded_file:
        dicom_bytes = uploaded_file.read()
        ds, img = dicom_to_image(dicom_bytes)
        st.image(img, caption="Original Image", use_column_width=True, clamp=True)

with col2:
    st.subheader("🛡️ Anonymized Output")
    if uploaded_file and st.button("🔍 Detect and Mask"):
        st.info("Running detection...")
        ocr = load_ocr()
        result = ocr.readtext(img, details=True)

        boxes = result[0]["boundary_result"] if result and "boundary_result" in result[0] else []
        if boxes:
            masked_img = mask_text(img, boxes, draw_boxes=show_boxes)
            st.image(masked_img, caption="Masked Image", use_column_width=True, clamp=True)
            masked_dicom_bytes = save_masked_dicom(ds, masked_img)

            st.download_button(
                label="💾 Download Masked DICOM",
                data=masked_dicom_bytes,
                file_name="masked.dcm",
                mime="application/dicom"
            )
        else:
            st.warning("No text detected.")

