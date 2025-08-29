import streamlit as st
import requests
from PIL import Image, ImageDraw

API_URL = "https://minetrack-6xv1.onrender.com/predict/thermal"

st.title("Détection de Mines sur Image Thermique")
st.markdown("Envoyez une image thermique, et recevez les prédictions du modèle YOLOv8.")

uploaded_file = st.file_uploader("Choisissez une image thermique (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Image originale", use_container_width=True)

    if st.button("Analyser l'image"):
        with st.spinner("Analyse en cours..."):
            files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
            try:
                response = requests.post(API_URL, files=files)
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                st.error(f"Erreur lors de l'envoi à l'API : {e}")
            else:
                result = response.json()
                detections = result.get("detections", [])

                # Dessiner les boîtes sur l'image
                image = Image.open(uploaded_file).convert("RGB")
                draw = ImageDraw.Draw(image)

                for det in detections:
                    bbox = det["bbox"]
                    confidence = det["confidence"]
                    draw.rectangle(bbox, outline="red", width=3)
                    draw.text((bbox[0], bbox[1] - 10), f"{confidence:.2f}", fill="red")

                st.image(image, caption="Image avec détections", use_column_width=True)
                st.success(f"{len(detections)} détection(s) trouvée(s)")
