import torch
import io
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights

from app.domain.errors import InvalidImageError
from app.domain.models import DetectedBox, BoundingBox
from PIL import Image, ImageOps

from app.images import open_image


class TorchDetector:
    def __init__(
        self,
        weights_path: str | None = None,
        confidence_threshold: float = 0.5,
        device: str = "cpu",
    ):
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self._categories = weights.meta["categories"]
        self._transform = weights.transforms()
        self._confidence_threshold = confidence_threshold
        self._device = torch.device(device)

        if weights_path is None:
            self._model = fasterrcnn_resnet50_fpn_v2(weights=weights)
        else:
            self._model = fasterrcnn_resnet50_fpn_v2(weights=None)
            self._model.load_state_dict(torch.load(weights_path, map_location=self._device))

        self._model.to(self._device)
        self._model.eval()

    def get_categories(self) -> list[str]:
        return [name for name in self._categories if name not in ("__background__", "N/A")]

    def detect(self, image_bytes: bytes) -> list[DetectedBox]:
        image = open_image(image_bytes).image.convert("RGB")
        input_tensor = self._transform(image).to(self._device)

        with torch.no_grad():
            predictions = self._model([input_tensor])[0]

        detected_boxes = []
        for box, label, score in zip(
                predictions["boxes"], predictions["labels"], predictions["scores"]):
            if score < self._confidence_threshold:
                continue
            x_min, y_min, x_max, y_max = box.tolist()
            detected_boxes.append(
                DetectedBox(
                    box=BoundingBox(
                        x=x_min,
                        y=y_min,
                        width=x_max - x_min,
                        height=y_max - y_min,
                    ),
                    label=self._categories[label],
                    confidence=score.item(),
                )
            )
        return detected_boxes