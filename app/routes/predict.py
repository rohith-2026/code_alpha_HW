from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.services.image_processor import BlankImageError
from app.services.predictor import predict_character


router = APIRouter()


@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
    recognition_mode: str = Form("all"),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload must be an image file.")

    image_bytes = await file.read()
    try:
        return predict_character(image_bytes, recognition_mode=recognition_mode)
    except BlankImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
