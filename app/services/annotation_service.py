import io
import json
from pathlib import Path

from PIL import Image
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights

from app.config import ANNOTATIONS_FILE, IMAGES_DIR
from app.domain.models import ExpertCorrectedBox

_INITIAL_CATEGORIES = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT.meta["categories"]
_INITIAL_LABEL_TO_ID = {
    name: idx for idx, name in enumerate(_INITIAL_CATEGORIES) if name not in (None, "__background__", "N/A")
}


class AnnotationService:
    def __init__(self, images_dir: str = IMAGES_DIR, annotations_file: str = ANNOTATIONS_FILE):
        self._images_dir = Path(images_dir)
        self._annotations_file = Path(annotations_file)
        self._images_dir.mkdir(parents=True, exist_ok=True)
        self._annotations_file.parent.mkdir(parents=True, exist_ok=True)

    def save(self, image_bytes: bytes, file_name: str, boxes: list[ExpertCorrectedBox]) -> None:
        data = self._load()

        image = Image.open(io.BytesIO(image_bytes))
        width, height = image.size

        image_id = self._next_id(data["images"])
        next_annotation_id = self._next_id(data["annotations"])

        stored_file_name = f"{image_id}_{file_name}"
        (self._images_dir / stored_file_name).write_bytes(image_bytes)

        data["images"].append({
            "id": image_id,
            "file_name": stored_file_name,
            "width": width,
            "height": height,
        })

        for offset, box in enumerate(boxes):
            category_id = self._get_or_create_category_id(data["categories"], box.label)
            data["annotations"].append({
                "id": next_annotation_id + offset,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [box.box.x, box.box.y, box.box.width, box.box.height],
                "area": box.box.width * box.box.height,
                "iscrowd": 0,
            })

        self._save(data)

    @staticmethod
    def _get_or_create_category_id(categories: list[dict], label: str) -> int:
        for category in categories:
            if category["name"] == label:
                return category["id"]

        new_id = max((c["id"] for c in categories), default=-1) + 1
        categories.append({"id": new_id, "name": label})
        return new_id

    def _load(self) -> dict:
        if not self._annotations_file.exists():
            return {
                "images": [],
                "annotations": [],
                "categories": [
                    {"id": category_id, "name": name}
                    for name, category_id in _INITIAL_LABEL_TO_ID.items()
                ],
            }
        return json.loads(self._annotations_file.read_text())

    def _save(self, data: dict) -> None:
        self._annotations_file.write_text(json.dumps(data, indent=2))

    @staticmethod
    def _next_id(items: list[dict]) -> int:
        if not items:
            return 1
        return max(item["id"] for item in items) + 1