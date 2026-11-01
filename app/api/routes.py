from secrets import compare_digest

from fastapi import APIRouter, UploadFile, Depends, Form, HTTPException, Header
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from app.repositories.active_model import save_active_model_to_json, ActiveModel
from app.config import MODEL_HOT_UPDATE_SECRET, ACTIVE_MODEL_PATH
from app.detectors.create_detector import create_detector
from app.domain.models import DetectedBox, AnnotationPayload, ClassInfo, ModelUpdatePayload
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


def require_model_update_secret(x_model_secret: str | None = Header(default=None)) -> None:
    if (
            MODEL_HOT_UPDATE_SECRET is None
            or x_model_secret is None
            or not compare_digest(MODEL_HOT_UPDATE_SECRET, x_model_secret)
    ):
        raise HTTPException(status_code=403, detail="Invalid or missing model update secret.")


@router.put("/model", dependencies=[Depends(require_model_update_secret)])
def model_hot_update(
        payload: ModelUpdatePayload,
        detection_service: DetectionService = Depends(get_detection_service),
        annotation_service: AnnotationService = Depends(get_annotations_service),
) -> dict:
    categories = {category.id: category.name for category in payload.categories}
    active_model = ActiveModel(weights_path=payload.weights_path, categories=categories)
    detector = create_detector(active_model)
    annotation_service.register_model_categories(categories)
    save_active_model_to_json(ACTIVE_MODEL_PATH, active_model)
    detection_service.replace_detector(detector)
    annotation_service.delete_images(payload.trained_image_ids)
    return {"status": "success"}
