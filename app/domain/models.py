from typing import Annotated

from pydantic import BaseModel, StringConstraints, Field


class BoundingBox(BaseModel):
    x: float = Field(ge=0)
    y: float = Field(ge=0)
    width: float = Field(gt=0)
    height: float = Field(gt=0)

class DetectedBox(BaseModel):
    box: BoundingBox
    label: str
    confidence: float

class ExpertCorrectedBox(BaseModel):
    box: BoundingBox
    label: Annotated[str, StringConstraints(strip_whitespace=True, to_lower=True, min_length=1)]

class AnnotationPayload(BaseModel):
    detections: list[ExpertCorrectedBox]

class ClassInfo(BaseModel):
    name: str
    learned: bool