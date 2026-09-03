from fastapi import APIRouter, UploadFile, Depends
from app.domain.models import DetectedBox
from app.services.detection_service import DetectionService

router = APIRouter()

def get_detection_service() -> DetectionService:
    raise NotImplementedError("DetectionService is required.")

@router.post("/detect")
async def detect(
    image: UploadFile,
    detection_service: DetectionService = Depends(get_detection_service),
) -> list[DetectedBox]:
    image_bytes = await image.read()
    return detection_service.detection(image_bytes)