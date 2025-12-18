import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import cv2
from ultralytics import YOLO
import os

st.set_page_config(
    page_title="Minetrack - Détection de Mines",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Minetrack - Détection de Mines")
st.markdown("### Analyse d'images thermiques avec YOLOv8")
st.markdown("Téléchargez une image thermique pour détecter la présence de mines terrestres.")

# Charger le modèle au démarrage (avec cache)
@st.cache_resource
def load_model():
    try:
        model = YOLO("app/models/best.pt")
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle: {e}")
        return None

def equalize_histogram(pil_img):
    img_gray = np.array(pil_img.convert("L"))
    img_eq = cv2.equalizeHist(img_gray)
    img_eq_rgb = cv2.cvtColor(img_eq, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(img_eq_rgb)

def preprocess_image(image):
    image = image.convert("RGB")
    image = equalize_histogram(image)
    return image

def predict(model, image):
    results = model(image)
    detections = []
    
    for box in results[0].boxes:
        detection = {
            "class_id": int(box.cls.item()),
            "confidence": float(box.conf.item()),
            "bbox": [float(coord.item()) for coord in box.xyxy[0]]
        }
        detections.append(detection)
    
    return detections

# Interface
with st.sidebar:
    st.markdown("### ℹ️ À propos")
    st.markdown("Application de détection de mines terrestres utilisant YOLOv8 sur images thermiques.")
    
    model = load_model()
    if model:
        st.success("✅ Modèle chargé")
    else:
        st.error("❌ Modèle non disponible")

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader(
        "📁 Choisissez une image thermique", 
        type=["jpg", "jpeg", "png"],
        help="Formats acceptés: JPG, JPEG, PNG"
    )
    
    if uploaded_file:
        st.image(uploaded_file, caption="Image originale", use_container_width=True)

with col2:
    if uploaded_file and model:
        if st.button("🔍 Analyser l'image", type="primary", use_container_width=True):
            with st.spinner("Analyse en cours..."):
                try:
                    # Charger et prétraiter l'image
                    image = Image.open(uploaded_file)
                    processed_image = preprocess_image(image)
                    
                    # Prédiction
                    detections = predict(model, processed_image)
                    
                    # Dessiner les résultats
                    display_image = image.convert("RGB")
                    draw = ImageDraw.Draw(display_image)

                    for det in detections:
                        bbox = det["bbox"]
                        confidence = det["confidence"]
                        draw.rectangle(bbox, outline="red", width=3)
                        draw.text((bbox[0], bbox[1] - 10), f"{confidence:.2f}", fill="red")

                    st.image(display_image, caption="Résultat de l'analyse", use_container_width=True)
                    
                    if len(detections) > 0:
                        st.error(f"⚠️ {len(detections)} mine(s) détectée(s)")
                        
                        with st.expander("Détails des détections"):
                            for i, det in enumerate(detections, 1):
                                st.write(f"**Mine {i}** - Confiance: {det['confidence']:.2%}")
                                st.write(f"Position: {det['bbox']}")
                    else:
                        st.success("✅ Aucune mine détectée")
                        
                except Exception as e:
                    st.error(f"❌ Erreur: {str(e)}")
                    st.exception(e)

st.divider()
st.markdown("**Note:** Cette application utilise un modèle YOLOv8 entraîné sur des images thermiques.")

