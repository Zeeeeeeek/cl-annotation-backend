from datetime import datetime, UTC

from sqlalchemy import CheckConstraint, event
from sqlmodel import SQLModel, Field, create_engine
from pathlib import Path


def utc_now() -> datetime:
    return datetime.now(UTC)

class Category(SQLModel, table=True):
    __tablename__ = "categories"
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(unique=True, nullable=False)


class ImageRecord(SQLModel, table=True):
    __tablename__ = "images"
    id: int | None = Field(default=None, primary_key=True)
    sha256: str = Field(unique=True, nullable=False)
    file_name: str
    original_name: str
    width: int
    height: int
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class Annotation(SQLModel, table=True):
    __tablename__ = "annotations"
    __table_args__ = (
        CheckConstraint("x >= 0"),
        CheckConstraint("y >= 0"),
        CheckConstraint("width > 0"),
        CheckConstraint("height > 0"),
    )
    id: int | None = Field(default=None, primary_key=True)
    image_id: int = Field(foreign_key="images.id", nullable=False, ondelete="CASCADE")
    category_id: int = Field(foreign_key="categories.id", nullable=False)
    x: float = Field(nullable=False)
    y: float = Field(nullable=False)
    width: float = Field(nullable=False)
    height: float = Field(nullable=False)


def create_db_engine(db_path: str):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}", echo=False, connect_args={"check_same_thread": False})

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(dbapi_connection, _connection_record):
        dbapi_connection.execute("PRAGMA foreign_keys = ON")

    SQLModel.metadata.create_all(engine)
    return engine