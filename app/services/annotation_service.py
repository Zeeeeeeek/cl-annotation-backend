import hashlib
from pathlib import Path

from app.config import IMAGES_DIR
from app.database import ImageRecord
from app.domain.errors import InvalidAnnotationError
from app.domain.models import ExpertCorrectedBox
from app.images import open_image, OpenedImage
from app.repositories.annotation_repository import AnnotationRepository


BOX_TOLERANCE = 0.01

class AnnotationService:
    def __init__(self, repository: AnnotationRepository, initial_categories: dict[int, str],
                 images_dir: str = IMAGES_DIR):
        self._repository = repository
        self._images_dir = Path(images_dir)
        self._images_dir.mkdir(parents=True, exist_ok=True)
        self._repository.seed_categories(initial_categories)

    def save(self, image_bytes: bytes, file_name: str, boxes: list[ExpertCorrectedBox]) -> None:
        opened = open_image(image_bytes)
        width, height = opened.image.size
        self._check_boxes_inside(boxes, width, height)

        sha256 = hashlib.sha256(image_bytes).hexdigest()
        stored_file_name = f"{sha256}{Path(file_name).suffix.lower()}"
        self._store_file(stored_file_name, image_bytes, opened)

        self._repository.replace_image_annotations(
            ImageRecord(sha256=sha256, file_name=stored_file_name, original_name=file_name, width=width, height=height),
            boxes,
        )

    def get_categories(self) -> list[str]:
        return self._repository.get_category_names()

    @staticmethod
    def _check_boxes_inside(boxes: list[ExpertCorrectedBox], width: int, height: int) -> None:
        for index, box in enumerate(boxes):
            right = box.box.x + box.box.width
            bottom = box.box.y + box.box.height
            if right > width + BOX_TOLERANCE or bottom > height + BOX_TOLERANCE:
                raise InvalidAnnotationError(f"Box {index} is outside the image ({width}x{height})")

    def _store_file(self, stored_file_name: str, image_bytes: bytes, opened: OpenedImage) -> None:
        image_path = self._images_dir / stored_file_name
        if image_path.exists():
            return
        if opened.was_rotated:
            opened.image.save(image_path, format=opened.format, quality=95)
        else:
            image_path.write_bytes(image_bytes)
