from fastapi import FastAPI

from app.api.routes import router, get_detection_service, get_annotations_service
from app.detectors.TorchDetector import TorchDetector
from app.services.annotation_service import AnnotationService
from app.services.detection_service import DetectionService
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

origins = [
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = TorchDetector(device = "cuda")
detection_service = DetectionService(detector)
annotation_service = AnnotationService()

app.dependency_overrides[get_detection_service] = lambda: detection_service
app.dependency_overrides[get_annotations_service] = lambda: annotation_service

app.include_router(router)