import logging

from fastapi import FastAPI

from app.active_model import load_active_model_from_json
from app.api.web_app import create_web_app
from app.config import PORT, DATABASE_PATH, ACTIVE_MODEL_PATH
from app.database import create_db_engine
from app.detectors.create_detector import create_detector
from app.repositories.annotation_repository import AnnotationRepository
from app.services.annotation_service import AnnotationService
from app.services.detection_service import DetectionService

import uvicorn

logger = logging.getLogger(__name__)


def build_app() -> FastAPI:
    logging.basicConfig(level=logging.INFO)
    active_model = load_active_model_from_json(ACTIVE_MODEL_PATH)
    detector = create_detector(active_model)

    logger.info("Loaded model: %s", active_model.weights_path if active_model else "default torchvision weights")

    detection_service = DetectionService(detector)
    db_engine = create_db_engine(DATABASE_PATH)
    annotation_service = AnnotationService(AnnotationRepository(db_engine),
                                           initial_categories=detector.get_categories_and_ids())
    return create_web_app(detection_service, annotation_service)


def main():
    uvicorn.run(build_app(), host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    main()
