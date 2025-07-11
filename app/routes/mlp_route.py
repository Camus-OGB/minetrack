from fastapi import APIRouter
from pydantic import BaseModel
from app.services.mlp_predict import predict_mlp

router = APIRouter()

class MLPInput(BaseModel):
    V: float
    H: float
    Soil_1: float
    Soil_2: float
    Soil_3: float
    Soil_4: float
    Soil_5: float
    Soil_6: float

@router.post("/mlp")
def predict_mlp_route(input: MLPInput):
    features = [
        input.V, input.H,
        input.Soil_1, input.Soil_2, input.Soil_3,
        input.Soil_4, input.Soil_5, input.Soil_6
    ]
    score = predict_mlp(features)
    return {
        "inputs": features,
        "mlp_score": score
    }
