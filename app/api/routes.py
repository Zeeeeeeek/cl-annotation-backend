from fastapi import APIRouter, UploadFile, Depends, Form, HTTPException
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from app.domain.models import DetectedBox, AnnotationPayload, ClassInfo
from app.services.annotation_service import AnnotationService
from app.services.detection_service import DetectionService

router = APIRouter()

def get_detection_service() -> DetectionService:
    raise NotImplementedError("DetectionService is required.")

def get_annotations_service() -> AnnotationService:
    raise NotImplementedError("AnnotationService is required.")

@router.post("/detect")
async def detect(
    image: UploadFile,
    detection_service: DetectionService = Depends(get_detection_service),
) -> list[DetectedBox]:
    image_bytes = await image.read()
    return detection_service.detection(image_bytes)

@router.post("/annotations")
async def save_annotations(
    image: UploadFile,
    annotations: str = Form(...),
    annotation_service: AnnotationService = Depends(get_annotations_service),
) -> dict:
    if image.filename is None:
        raise HTTPException(status_code=400, detail="Image file is required.")
    image_bytes = await image.read()
    try:
        payload = AnnotationPayload.model_validate_json(annotations)
    except ValidationError as error:
        raise RequestValidationError(error.errors())
    annotation_service.save(image_bytes, image.filename, payload.detections)
    return {"status": "success"}

@router.get("/classes")
async def get_classes(
    annotation_service: AnnotationService = Depends(get_annotations_service),
    detection_service: DetectionService = Depends(get_detection_service),
) -> list[ClassInfo]:
    learned = set(detection_service.get_learned_categories())
    return [ClassInfo(name=name, learned=name in learned) for name in annotation_service.get_categories()]

@router.get("/health")
async def health():
    return ""