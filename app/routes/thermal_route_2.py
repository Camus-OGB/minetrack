
# from fastapi import APIRouter,Form, UploadFile, File, FastAPI
# import uvicorn
# from app.services.thermal_predict import thermal_prediction

# router = APIRouter()
# app = FastAPI()

# @router.post("/thermal")
# async def predict_thermal(file: UploadFile = File(...)):
#     """
#     Endpoint for thermal image prediction using YOLOv8 model.
#     """
#     if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
#         return {"error": "Invalid file format. Please upload a JPG or PNG image."}
#     try:
#         result = await thermal_prediction(file)
#         return result
#     except Exception as e:
#         return {"error": str(e)}
    
# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=8002)
