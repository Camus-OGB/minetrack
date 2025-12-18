import streamlit as st
import requests
from PIL import Image, ImageDraw
import os

API_URL = os.getenv("API_URL", "https://minetrack-6xv1.onrender.com/predict/thermal")

st.set_page_config(
    page_title="Minetrack - Détection de Mines",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Minetrack - Détection de Mines")
st.markdown("### Analyse d'images thermiques avec YOLOv8")
st.markdown("Téléchargez une image thermique pour détecter la présence de mines terrestres.")

# Test de connexion à l'API
with st.sidebar:
    st.markdown("### 🔌 État de l'API")
    try:
        health_response = requests.get(API_URL.replace("/predict/thermal", "/health"), timeout=10)
        if health_response.status_code == 200:
            st.success("✅ API connectée")
        else:
            st.warning("⚠️ API inaccessible")
    except:
        st.error("❌ API hors ligne")
        st.caption(f"URL: {API_URL}")
        st.info("💡 Si l'API est sur Render (gratuit), elle peut prendre 30-60s à démarrer après inactivité.")

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
    if uploaded_file:
        if st.button("🔍 Analyser l'image", type="primary", use_container_width=True):
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    if attempt > 0:
                        st.info(f"🔄 Nouvelle tentative ({attempt + 1}/{max_retries})...")
                    
                    with st.spinner("Analyse en cours... (peut prendre 30-90s si l'API démarre)"):
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                        response = requests.post(API_URL, files=files, timeout=90)
                        response.raise_for_status()
                        
                        result = response.json()
                        detections = result.get("detections", [])

                        image = Image.open(uploaded_file).convert("RGB")
                        draw = ImageDraw.Draw(image)

                        for det in detections:
                            bbox = det["bbox"]
                            confidence = det["confidence"]
                            draw.rectangle(bbox, outline="red", width=3)
                            draw.text((bbox[0], bbox[1] - 10), f"{confidence:.2f}", fill="red")

                        st.image(image, caption="Résultat de l'analyse", use_container_width=True)
                        
                        if len(detections) > 0:
                            st.error(f"⚠️ {len(detections)} mine(s) détectée(s)")
                            
                            with st.expander("Détails des détections"):
                                for i, det in enumerate(detections, 1):
                                    st.write(f"**Mine {i}** - Confiance: {det['confidence']:.2%}")
                                    st.write(f"Position: {det['bbox']}")
                        else:
                            st.success("✅ Aucune mine détectée")
                        break
                        
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 502 and attempt < max_retries - 1:
                        st.warning("⏳ L'API démarre, nouvelle tentative dans 5 secondes...")
                        import time
                        time.sleep(5)
                        continue
                    else:
                        st.error(f"❌ Erreur serveur: {e.response.status_code}")
                        st.info("💡 L'API Render est peut-être en train de redémarrer. Réessayez dans 30 secondes.")
                        break
                except requests.exceptions.Timeout:
                    st.error("❌ Timeout: L'API met trop de temps à répondre")
                    st.info("💡 Si c'est la première requête, l'API Render peut mettre jusqu'à 1 minute à démarrer.")
                    break
                except requests.exceptions.RequestException as e:
                    st.error(f"❌ Erreur de connexion: {str(e)}")
                    break
                except Exception as e:
                    st.error(f"❌ Erreur: {str(e)}")
                    break

st.divider()
st.markdown("**Note:** Cette application utilise un modèle YOLOv8 entraîné sur des images thermiques.")

