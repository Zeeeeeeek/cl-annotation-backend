import hashlib
from pathlib import Path

from app.config import IMAGES_DIR
from app.database import ImageRecord
from app.domain.errors import InvalidAnnotationError
from app.domain.models import ExpertCorrectedBox
from app.images import open_image, OpenedImage
from app.repositories.annotation_repository import AnnotationRepository


BOX_TOLERANCE = 0.01


def _check_categories_match(model_categories: dict[int, str], db_categories: dict[int, str]) -> None:
    db_ids_by_name = {name: category_id for category_id, name in db_categories.items()}
    for category_id, name in model_categories.items():
        if category_id in db_categories and db_categories[category_id] != name:
            raise ValueError(
                f"Category id {category_id} is '{name}' in the model but '{db_categories[category_id]}' in the database")
        if name in db_ids_by_name and db_ids_by_name[name] != category_id:
            raise ValueError(
                f"Category '{name}' has id {category_id} in the model but {db_ids_by_name[name]} in the database")

class AnnotationService:
    def __init__(self, repository: AnnotationRepository, initial_categories: dict[int, str],
                 images_dir: str = IMAGES_DIR):
        self._repository = repository
        self._images_dir = Path(images_dir)
        self._images_dir.mkdir(parents=True, exist_ok=True)
        _check_categories_match(initial_categories, self._repository.get_categories_by_id())
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

    def delete_images(self, image_ids: list[int]) -> None:
        file_names = self._repository.delete_images(image_ids)
        for file_name in file_names:
            (self._images_dir / file_name).unlink(missing_ok=True)

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
