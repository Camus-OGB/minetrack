from fastapi import APIRouter, UploadFile, File
from app.services.thermal_predict import thermal_prediction
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/thermal")
async def predict_thermal(file: UploadFile = File(...)):
    """
    Endpoint for thermal image prediction using YOLOv8 model.
    """
    logger.info(f"Requête reçue pour fichier: {file.filename}")
    
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        logger.warning(f"Format invalide: {file.filename}")
        return {"error": "Invalid file format. Please upload a JPG or PNG image."}
    
    try:
        result = await thermal_prediction(file)
        logger.info(f"Prédiction réussie: {result['num_detections']} détections")
        return result
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {str(e)}", exc_info=True)
        return {"error": str(e)}

