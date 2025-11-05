import streamlit as st
import numpy as np
from joblib import load
from skimage.feature import hog
from PIL import Image, ImageOps


@st.cache_resource
def load_models():
    try:
        # Import YOLO here so missing system libs (libGL) don't crash at module import time.
        from ultralytics import YOLO
        facemodel = YOLO("yolov8m-face.pt")
    except Exception as e:
        # Return None for facemodel and leave the exception to be shown in the UI later.
        facemodel = None
        load_models._error = e  # attach error to function for UI reporting

    svm_model = None
    try:
        svm_model = load("svm_model.joblib")
    except Exception:
        # If the SVM model is missing, keep None and report in UI later.
        svm_model = None

    return facemodel, svm_model


facemodel, svm_clf_loaded = load_models()


def extract_hog_features(image):
    features = hog(image,
                   orientations=9,
                   pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2),
                   block_norm='L2-Hys',
                   transform_sqrt=True)
    return features


st.title("Emotion Detection App")

uploaded_file = st.file_uploader("Upload a face image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img = ImageOps.exif_transpose(img)  # Handle image orientation

    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Convert image to grayscale and resize for processing
    gray_img = img.convert("L")
    results = facemodel(np.array(img), stream=True)

    faces_48x48 = []
    boxes = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            face = gray_img.crop((x1, y1, x2, y2)).resize((48, 48))
            faces_48x48.append(np.array(face))
            boxes.append((x1, y1, x2, y2))

    if len(faces_48x48) > 0:
        features_list = [extract_hog_features(f) for f in faces_48x48]
        X_test = np.array(features_list)
        y_pred = svm_clf_loaded.predict(X_test)

        for (x1, y1, x2, y2), emo in zip(boxes, y_pred):
            st.text(f"Face at ({x1}, {y1}, {x2}, {y2}): {emo}")

        st.success(f"Detected {len(y_pred)} face(s)")
    else:
        st.warning("No faces detected.")
