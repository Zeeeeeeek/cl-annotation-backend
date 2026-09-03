from fastapi import FastAPI

from app.api.routes import router, get_detection_service
from app.detectors.TorchDetector import TorchDetector
from app.services.detection_service import DetectionService

app = FastAPI()

detector = TorchDetector(device = "cuda")
detection_service = DetectionService(detector)

app.dependency_overrides[get_detection_service] = lambda: detection_service

app.include_router(router)