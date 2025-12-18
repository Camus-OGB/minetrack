# 🔍 Minetrack - Détection de Mines

Système de détection de mines terrestres utilisant des images thermiques et des données magnétiques avec YOLOv8 et des réseaux de neurones.

## 📋 Description

Ce projet combine deux approches pour la détection de mines :
- **Détection Thermique** : YOLOv8 entraîné sur images thermiques
- **Analyse Magnétique** : Réseau MLP pour classifier à partir de données magnétiques

## 🚀 Installation


### Prérequis

- Python 3.9+
- pip

### Setup

```bash
git clone https://github.com/akiragithub/minetrack.git
cd minetrack

python -m venv venv
venv\Scripts\activate  # Windows

pip install -r requirements.txt

cp .env.example .env
```

## 💻 Utilisation

### API FastAPI

```bash
uvicorn app.main:app --reload
```

API disponible sur `http://localhost:8000`  
Documentation : `http://localhost:8000/docs`

### Interface Streamlit

```bash
streamlit run streamlit_app.py
```

Interface sur `http://localhost:8501`

## 📖 API

### POST /predict/thermal

Détection sur image thermique

**Request:** Image JPG/PNG  
**Response:**
```json
{
  "num_detections": 2,
  "detections": [
    {
      "class_id": 0,
      "confidence": 0.87,
      "bbox": [120.5, 230.1, 180.3, 290.7]
    }
  ],
  "confidence_max": 0.87
}
```

### POST /predict/mlp

Analyse magnétique

**Request:**
```json
{
  "V": 1.5, "H": 2.3,
  "Soil_1": 0.8, "Soil_2": 0.6,
  "Soil_3": 0.9, "Soil_4": 0.7,
  "Soil_5": 0.5, "Soil_6": 0.4
}
```

**Response:**
```json
{
  "inputs": [...],
  "mlp_score": 0.92
}
```

## 🌐 Déploiement

### Streamlit Cloud

1. Push sur GitHub
2. Connecter le repo sur [streamlit.io/cloud](https://streamlit.io/cloud)
3. Sélectionner `streamlit_app.py`
4. Configurer `API_URL` dans les secrets

### API (Render)

1. Créer un Web Service sur [render.com](https://render.com)
2. Build: `pip install -r requirements.txt`
3. Start: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`


## 📁 Structure

```
minetrack/
├── app/
│   ├── main.py              # API FastAPI
│   ├── routes/              # Endpoints
│   ├── services/            # Logique métier
│   ├── models/              # Modèles ML
│   └── notebooks/           # Notebooks Jupyter
├── streamlit_app.py         # Interface web
├── requirements.txt         
└── .env.example
```

## 🛠️ Technologies

- FastAPI, Uvicorn
- PyTorch, YOLOv8 (Ultralytics)
- scikit-learn
- Streamlit
- OpenCV, Pillow
- NumPy, Pandas

---

Projet personnel - [@akiragithub](https://github.com/akiragithub)


