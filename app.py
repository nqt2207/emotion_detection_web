import streamlit as st
import cv2
import cvzone
import numpy as np
from PIL import Image
from joblib import load
from ultralytics import YOLO
from skimage.feature import hog

# ======================
# Load mô hình
# ======================
@st.cache_resource
def load_models():
    facemodel = YOLO("yolov8m-face.pt")
    svm_model = load("svm_model.joblib")
    return facemodel, svm_model

facemodel, svm_clf_loaded = load_models()

# ======================
# Hàm trích xuất đặc trưng HOG
# ======================
def extract_hog_features(image):
    features = hog(image,
                   orientations=9,
                   pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2),
                   block_norm='L2-Hys',
                   transform_sqrt=True)
    return features

# ======================
# Giao diện web
# ======================
st.title("🎭 Nhận diện cảm xúc bằng YOLO + SVM (HOG)")

uploaded_file = st.file_uploader("Tải lên ảnh khuôn mặt...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Đọc ảnh
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    if img is None:
        st.error("Không thể đọc ảnh, vui lòng thử lại.")
    else:
        # Hiển thị ảnh gốc
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Ảnh gốc", use_column_width=True)

        # Phát hiện khuôn mặt
        results = facemodel(img, stream=True)

        faces_48x48 = []
        boxes = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face = img[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (48, 48))
                faces_48x48.append(resized)
                boxes.append((x1, y1, x2, y2))

        # Nếu phát hiện được khuôn mặt
        if len(faces_48x48) > 0:
            features_list = [extract_hog_features(f) for f in faces_48x48]
            X_test = np.array(features_list)
            y_pred = svm_clf_loaded.predict(X_test)

            for (x1, y1, x2, y2), emo in zip(boxes, y_pred):
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, str(emo), (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            st.success(f"✅ Phát hiện {len(y_pred)} khuôn mặt")
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                     caption="Kết quả dự đoán",
                     use_column_width=True)
        else:
            st.warning("❌ Không phát hiện được khuôn mặt nào.")
