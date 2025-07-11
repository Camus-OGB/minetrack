import streamlit as st
import requests
from PIL import Image, ImageDraw
import io

API_URL = "http://localhost:8000/predict/thermal"

st.title("Détection de Mines sur Image Thermique")
st.markdown("Envoyez une image thermique, et recevez les prédictions du modèle YOLOv8.")

uploaded_file = st.file_uploader("Choisissez une image thermique (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Image originale", use_container_width=True)

    if st.button("Analyser l'image"):
        with st.spinner("Analyse en cours..."):
            files = {"file": uploaded_file.getvalue()}
            response = requests.post(API_URL, files={"file": uploaded_file})
        
        if response.status_code == 200:
            result = response.json()
            detections = result["detections"]

            # Charger l'image et dessiner les boîtes
            image = Image.open(uploaded_file).convert("RGB")
            draw = ImageDraw.Draw(image)

            for det in detections:
                bbox = det["bbox"]
                confidence = det["confidence"]
                class_id = det["class_id"]

                # Dessiner la boîte
                draw.rectangle(bbox, outline="red", width=3)
                draw.text((bbox[0], bbox[1] - 10), f"{confidence:.2f}", fill="red")

            st.image(image, caption="Image avec détections", use_column_width=True)
            st.success(f"{len(detections)} détection(s) trouvée(s)")
        else:
            st.error("Erreur lors de l'envoi à l'API : " + response.text)
