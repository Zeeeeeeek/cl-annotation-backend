from dataclasses import dataclass

from sqlalchemy import delete
from sqlmodel import Session, select, col

from app.repositories.database import Annotation, Category, ImageRecord, utc_now
from app.domain.models import ExpertCorrectedBox

@dataclass
class Dataset:
    images: list[ImageRecord]
    annotations: list[Annotation]
    categories: list[Category]

class AnnotationRepository:
    def __init__(self, engine):
        self._engine = engine

    def seed_categories(self, categories: dict[int, str]) -> None:
        with Session(self._engine) as session:
            existing_ids = set(session.exec(select(Category.id)).all())
            for category_id, name in categories.items():
                if category_id not in existing_ids:
                    session.add(Category(id=category_id, name=name))
            session.commit()

    def get_category_names(self) -> list[str]:
        with Session(self._engine) as session:
            categories = session.exec(select(Category).order_by(Category.id)).all()
            return [category.name for category in categories]

    def get_categories_by_id(self) -> dict[int, str]:
        with Session(self._engine) as session:
            categories = session.exec(select(Category)).all()
            return {category.id: category.name for category in categories if category.id is not None}

    def replace_image_annotations(self, image: ImageRecord, boxes: list[ExpertCorrectedBox]) -> None:
        with Session(self._engine) as session:
            stored = session.exec(select(ImageRecord).where(ImageRecord.sha256 == image.sha256)).first()
            if stored is None:
                session.add(image)
                session.flush()
                stored = image
            else:
                session.exec(delete(Annotation).where(Annotation.image_id == stored.id))
                stored.updated_at = utc_now()
            assert stored.id is not None
            for box in boxes:
                session.add(Annotation(
                    image_id=stored.id,
                    category_id=self._get_or_create_category_id(session, box.label),
                    x=box.box.x,
                    y=box.box.y,
                    width=box.box.width,
                    height=box.box.height,
                ))
            session.commit()

    def get_dataset(self) -> Dataset:
        with Session(self._engine) as session:
            images = list(session.exec(select(ImageRecord).order_by(ImageRecord.id)).all())
            annotations = list(session.exec(select(Annotation).order_by(Annotation.id)).all())
            categories = list(session.exec(select(Category).order_by(Category.id)).all())
        return Dataset(images, annotations, categories)

    def delete_images(self, image_ids: list[int]) -> list[str]:
        with Session(self._engine) as session:
            images = session.exec(select(ImageRecord).where(col(ImageRecord.id).in_(image_ids))).all()
            file_names = [image.file_name for image in images]
            for image in images:
                session.delete(image)
            session.commit()
        return file_names

    @staticmethod
    def _get_or_create_category_id(session: Session, label: str) -> int:
        category = session.exec(select(Category).where(Category.name == label)).first()
        if category is None:
            category = Category(name=label)
            session.add(category)
            session.flush()
        assert category.id is not None
        return category.id
