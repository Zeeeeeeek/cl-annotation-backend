from pydantic import BaseModel

class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float

class DetectedBox(BaseModel):
    box: BoundingBox
    label: str
    confidence: float

class ExpertCorrectedBox(BaseModel):
    box: BoundingBox
    label: str

class AnnotationPayload(BaseModel):
    detections: list[ExpertCorrectedBox]