from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import router, get_detection_service, get_annotations_service
from app.config import CORS_ORIGINS, PORT, DEVICE, CONFIDENCE_THRESHOLD, DATABASE_PATH
from app.database import create_db_engine
from app.detectors.TorchDetector import TorchDetector
from app.domain.errors import  InvalidInputError
from app.repositories.annotation_repository import AnnotationRepository
from app.services.annotation_service import AnnotationService
from app.services.detection_service import DetectionService
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"]
)

@app.exception_handler(InvalidInputError)
async def handle_invalid_input(_request: Request, error: InvalidInputError):
    return JSONResponse(status_code=422, content={"detail": str(error)})

detector = TorchDetector(device=DEVICE, confidence_threshold=CONFIDENCE_THRESHOLD)
detection_service = DetectionService(detector)
engine = create_db_engine(DATABASE_PATH)
annotation_service = AnnotationService(AnnotationRepository(engine))

app.dependency_overrides[get_detection_service] = lambda: detection_service
app.dependency_overrides[get_annotations_service] = lambda: annotation_service

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)