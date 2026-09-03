from app.detectors.TorchDetector import  TorchDetector

from app.domain.models import DetectedBox

class DetectionService:
    def __init__(self, detector: TorchDetector):
        self.detector = detector

    def detection(self, image_bytes: bytes) -> list[DetectedBox]:
        return self.detector.detect(image_bytes)