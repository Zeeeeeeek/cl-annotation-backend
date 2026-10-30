from sqlalchemy import delete
from sqlmodel import Session, select

from app.database import Annotation, Category, ImageRecord, utc_now
from app.domain.models import ExpertCorrectedBox


class AnnotationRepository:
    def __init__(self, engine):
        self._engine = engine

    def seed_categories(self, categories: dict[str, int]) -> None:
        with Session(self._engine) as session:
            existing_ids = set(session.exec(select(Category.id)).all())
            for name, category_id in categories.items():
                if category_id not in existing_ids:
                    session.add(Category(id=category_id, name=name))
            session.commit()

    def get_category_names(self) -> list[str]:
        with Session(self._engine) as session:
            categories = session.exec(select(Category).order_by(Category.id)).all()
            return [category.name for category in categories]

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

    @staticmethod
    def _get_or_create_category_id(session: Session, label: str) -> int:
        category = session.exec(select(Category).where(Category.name == label)).first()
        if category is None:
            category = Category(name=label)
            session.add(category)
            session.flush()
        assert category.id is not None
        return category.id
