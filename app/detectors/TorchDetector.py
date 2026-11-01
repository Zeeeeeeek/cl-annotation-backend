import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights

from app.domain.errors import InvalidModelError
from app.domain.models import DetectedBox, BoundingBox

from app.domain.images import open_image


def _to_index_list(categories: dict[int, str]) -> list[str]:
    names = ["N/A"] * (max(categories) + 1)
    names[0] = "__background__"
    for category_id, name in categories.items():
        names[category_id] = name
    return names


class TorchDetector:
    def __init__(
            self,
            weights_path: str | None = None,
            categories: dict[int, str] | None = None,
            confidence_threshold: float = 0.5,
            device: str = "cpu",
    ):
        default_weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self._transform = default_weights.transforms()
        self._confidence_threshold = confidence_threshold
        self._device = torch.device(device)

        if weights_path is None:
            self._categories = default_weights.meta["categories"]
            self._model = fasterrcnn_resnet50_fpn_v2(weights=default_weights)
        else:
            if categories is None:
                raise ValueError("categories are required when loading custom weights")
            self._categories = _to_index_list(categories)
            self._model = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None,
                                                     num_classes=len(self._categories))
            try:
                state_dict = torch.load(weights_path, map_location=self._device, weights_only=True)
                self._model.load_state_dict(state_dict)
            except Exception as error:
                raise InvalidModelError(f"Cannot load weights from {weights_path}: {error!r}") from error

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

    def get_categories_and_ids(self)-> dict[int, str]:
        return {
            category_id: category_name
            for category_id, category_name in enumerate(self._categories)
            if category_name not in ("__background__", "N/A")
        }