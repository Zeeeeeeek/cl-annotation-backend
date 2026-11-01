from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import get_detection_service, get_annotations_service, router
from app.config import CORS_ORIGINS
from app.domain.errors import InvalidInputError
from app.services.annotation_service import AnnotationService
from app.services.detection_service import DetectionService



async def handle_invalid_input(_request: Request, error: Exception):
    return JSONResponse(status_code=422, content={"detail": str(error)})

def create_web_app(detection_service: DetectionService, annotation_service: AnnotationService) -> FastAPI:
    web_app = FastAPI()
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"]
    )
    web_app.add_exception_handler(InvalidInputError, handle_invalid_input)
    web_app.dependency_overrides[get_detection_service] = lambda: detection_service
    web_app.dependency_overrides[get_annotations_service] = lambda: annotation_service
    web_app.include_router(router)
    return web_app