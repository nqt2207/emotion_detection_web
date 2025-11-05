import streamlit as st
import cv2
import numpy as np
from joblib import load
from ultralytics import YOLO
from skimage.feature import hog

# ======================
# Load models (cached)
# ======================
@st.cache_resource
def load_models():
    # Load YOLO face detector and pre-trained SVM classifier
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
# Web UI
# ======================
st.title("🎭 Nhận diện cảm xúc bằng YOLO + SVM (HOG)")

st.sidebar.markdown("### Cấu hình ứng dụng")
st.sidebar.write("Mô hình YOLO: yolov8m-face.pt")
st.sidebar.write("SVM HOG: svm_model.joblib")

uploaded_file = st.file_uploader("Tải lên ảnh khuôn mặt...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image bytes and decode with OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        st.error("Không thể đọc ảnh. Vui lòng thử file khác.")
    else:
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Ảnh gốc", use_column_width=True)

        # Detect faces with YOLO
        results = facemodel(img)

        faces_48x48 = []
        boxes = []

        for r in results:
            # r.boxes may be empty
            for box in r.boxes:
                xyxy = box.xyxy[0]
                x1, y1, x2, y2 = map(int, xyxy)
                # crop and validate
                face = img[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                if face.size == 0:
                    continue

                gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (48, 48))
                faces_48x48.append(resized)
                boxes.append((x1, y1, x2, y2))

        if len(faces_48x48) == 0:
            st.warning("❌ Không phát hiện được khuôn mặt nào.")
        else:
            features_list = [extract_hog_features(f) for f in faces_48x48]
            X_test = np.array(features_list)
            y_pred = svm_clf_loaded.predict(X_test)

            # draw boxes and labels
            for (x1, y1, x2, y2), emo in zip(boxes, y_pred):
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, str(emo), (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            st.success(f"✅ Phát hiện {len(y_pred)} khuôn mặt")
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Kết quả dự đoán", use_column_width=True)