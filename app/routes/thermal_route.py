from fastapi import APIRouter, UploadFile, File
from app.services.thermal_predict import thermal_prediction

router = APIRouter()

@router.post("/thermal")
async def predict_thermal(file: UploadFile = File(...)):
    """
    Endpoint for thermal image prediction using YOLOv8 model.
    """
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        return {"error": "Invalid file format. Please upload a JPG or PNG image."}
    try:
        result = await thermal_prediction(file)
        return result
    except Exception as e:
        return {"error": str(e)}
