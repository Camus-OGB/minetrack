from fastapi import FastAPI
from app.routes.thermal_route import router as thermal_route
from app.routes.mlp_route import router as mlp_route

app = FastAPI()

app.include_router(thermal_route, prefix="/predict", tags=["Thermal YOLOv8"])