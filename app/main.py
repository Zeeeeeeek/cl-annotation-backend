from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import router, get_detection_service, get_annotations_service
from app.detectors.TorchDetector import TorchDetector
from app.domain.errors import InvalidImageError, InvalidInputError
from app.services.annotation_service import AnnotationService
from app.services.detection_service import DetectionService
from fastapi.middleware.cors import CORSMiddleware
import os
import uvicorn

app = FastAPI()

origins = os.environ.get("CORS_ORIGINS", "http://localhost:5173").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"]
)

@app.exception_handler(InvalidInputError)
async def handle_invalid_input(_request: Request, error: InvalidInputError):
    return JSONResponse(status_code=422, content={"detail": str(error)})

detector = TorchDetector(device="cuda")
detection_service = DetectionService(detector)
annotation_service = AnnotationService()

app.dependency_overrides[get_detection_service] = lambda: detection_service
app.dependency_overrides[get_annotations_service] = lambda: annotation_service

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))