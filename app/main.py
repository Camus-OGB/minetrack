from fastapi import FastAPI
from app.routes.thermal_route import router as thermal_route
from app.routes.mlp_route import router as mlp_route
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Minetrack API",
    description="API de détection de mines terrestres utilisant YOLOv8 et MLP",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configuration CORS
origins = os.getenv("CORS_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if "*" in origins else origins,      # domaines autorisés
    allow_credentials=True,     # autoriser cookies et auth
    allow_methods=["*"],        # autoriser toutes les méthodes (GET, POST...)
    allow_headers=["*"],        # autoriser tous les headers
)

# Include routers
app.include_router(thermal_route, prefix="/predict", tags=["Thermal Detection"])
app.include_router(mlp_route, prefix="/predict", tags=["Magnetic Detection"])

@app.get("/", tags=["Health"])
async def root():
    """
    Point d'entrée de l'API - Health check
    """
    return {
        "message": "Minetrack API is running",
        "version": "1.0.0",
        "endpoints": {
            "thermal": "/predict/thermal",
            "magnetic": "/predict/mlp",
            "docs": "/docs"
        }
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """
    Vérification de l'état de l'API
    """
    return {"status": "healthy"}
