from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
import tempfile


def equalize_histogram(pil_img: Image.Image) -> Image.Image:
    img_gray = np.array(pil_img.convert("L"))
    img_eq = cv2.equalizeHist(img_gray)
    img_eq_rgb = cv2.cvtColor(img_eq, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(img_eq_rgb)

def preprocess_thermal_image(image_path: str) -> Image.Image:
    img = Image.open(image_path)
    img = img.convert("RGB")
    img = equalize_histogram(img)
    return img

# --- Charger le modèle thermique YOLOv8 (.pt) ---
yolo_model = YOLO("app/models/best.pt")

# --- Fonction principale de prédiction thermique ---
async def thermal_prediction(file):
    # Sauvegarder le fichier temporairement
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    # Prétraitement de l'image thermique
    image = preprocess_thermal_image(tmp_path)

    # Inférence avec YOLOv8
    results = yolo_model(image)

    # Extraction des détections sous forme de dictionnaire
    detections = []
    for box in results[0].boxes:
        detection = {
            "class_id": int(box.cls.item()),
            "confidence": float(box.conf.item()),
            "bbox": [float(coord.item()) for coord in box.xyxy[0]]
        }
        detections.append(detection)

    # Score max (utile pour une fusion tardive si besoin)
    max_score = max([d["confidence"] for d in detections], default=0.0)

    return {
        "num_detections": len(detections),
        "detections": detections,
        "confidence_max": max_score
    }