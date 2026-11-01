from app.detectors.TorchDetector import  TorchDetector

from app.domain.models import DetectedBox

class DetectionService:
    def __init__(self, detector: TorchDetector):
        self.detector = detector

    def detection(self, image_bytes: bytes) -> list[DetectedBox]:
        return self.detector.detect(image_bytes)

    def get_learned_categories(self) -> list[str]:
        return self.detector.get_categories()

    def replace_detector(self, detector: TorchDetector) -> None:
        self.detector = detector