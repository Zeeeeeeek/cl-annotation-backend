from app.repositories.active_model import ActiveModel
from app.config import CONFIDENCE_THRESHOLD, DEVICE
from app.detectors.TorchDetector import TorchDetector


def create_detector(active_model: ActiveModel | None) -> TorchDetector:
    if active_model is None:
        return TorchDetector(device=DEVICE, confidence_threshold=CONFIDENCE_THRESHOLD)
    return TorchDetector(
        weights_path=active_model.weights_path,
        categories=active_model.categories,
        device=DEVICE,
        confidence_threshold=CONFIDENCE_THRESHOLD,
    )