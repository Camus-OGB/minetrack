from fastapi import FastAPI
from app.routes.thermal_route import router as thermal_route
from app.routes.mlp_route import router as mlp_route
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Configuration CORS
origins = [
    "https://minetrack-bl8g6fdgkdb5khq8qrahzn.streamlit.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      # domaines autorisés
    allow_credentials=True,     # autoriser cookies et auth
    allow_methods=["*"],        # autoriser toutes les méthodes (GET, POST...)
    allow_headers=["*"],        # autoriser tous les headers
)

app.include_router(thermal_route, prefix="/predict", tags=["Thermal YOLOv8"])