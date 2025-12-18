# 🔍 Minetrack - Détection de Mines

Système de détection de mines terrestres utilisant des images thermiques avec YOLOv8.

## 📋 Description

Application Streamlit tout-en-un utilisant YOLOv8 pour détecter les mines sur des images thermiques.

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
```

## 💻 Utilisation

```bash
streamlit run streamlit_app.py
```

Application sur `http://localhost:8501`

## 🌐 Déploiement sur Streamlit Cloud

1. Push sur GitHub
2. Connecter le repo sur [streamlit.io/cloud](https://streamlit.io/cloud)
3. Sélectionner `streamlit_app.py`
4. Déployer !

## 📁 Structure

```
minetrack/
├── app/
│   ├── models/              # Modèles ML (best.pt)
│   └── notebooks/           # Notebooks Jupyter
├── streamlit_app.py         # Application principale
└── requirements.txt
```

## 🛠️ Technologies

- Streamlit
- PyTorch, YOLOv8 (Ultralytics)
- OpenCV, Pillow
- NumPy

---

Projet personnel - [@akiragithub](https://github.com/akiragithub)
