import hashlib
import io

import pytest
from PIL import Image

from app.database import create_db_engine
from app.domain.errors import InvalidAnnotationError, InvalidImageError, InvalidModelError
from app.domain.models import BoundingBox, ExpertCorrectedBox
from app.repositories.annotation_repository import AnnotationRepository
from app.services.annotation_service import AnnotationService

EXIF_ORIENTATION_TAG = 274


def make_jpeg(width=40, height=20, orientation=None):
    exif = Image.Exif()
    if orientation is not None:
        exif[EXIF_ORIENTATION_TAG] = orientation
    buffer = io.BytesIO()
    Image.new("RGB", (width, height)).save(buffer, "JPEG", exif=exif)
    return buffer.getvalue()


def make_box(x=0, y=0, width=10, height=10, label="person"):
    return ExpertCorrectedBox(box=BoundingBox(x=x, y=y, width=width, height=height), label=label)


@pytest.fixture
def repository(tmp_path):
    return AnnotationRepository(create_db_engine(str(tmp_path / "test.db")))


@pytest.fixture
def images_dir(tmp_path):
    return tmp_path / "images"


@pytest.fixture
def service(repository, images_dir):
    return AnnotationService(repository, initial_categories={1: "person"}, images_dir=str(images_dir))


def test_should_create_images_dir_when_service_is_created(service, images_dir):
    assert images_dir.is_dir()


def test_should_seed_initial_categories_when_service_is_created(service):
    assert service.get_categories() == ["person"]


def test_should_raise_when_initial_categories_conflict_with_database(repository, images_dir):
    repository.seed_categories({1: "human"})

    with pytest.raises(InvalidModelError):
        AnnotationService(repository, initial_categories={1: "person"}, images_dir=str(images_dir))


def test_should_store_file_named_after_hash_with_lowercase_extension(service, images_dir):
    image_bytes = make_jpeg()

    service.save(image_bytes, "Photo.JPG", [make_box()])

    stored = images_dir / f"{hashlib.sha256(image_bytes).hexdigest()}.jpg"
    assert stored.read_bytes() == image_bytes


def test_should_save_record_with_original_name_and_image_size(service, repository):
    service.save(make_jpeg(40, 20), "Photo.JPG", [make_box()])

    image = repository.get_dataset().images[0]
    assert (image.original_name, image.width, image.height) == ("Photo.JPG", 40, 20)


def test_should_save_image_without_annotations_when_boxes_are_empty(service, repository):
    service.save(make_jpeg(), "photo.jpg", [])

    dataset = repository.get_dataset()
    assert len(dataset.images) == 1
    assert dataset.annotations == []


def test_should_accept_box_when_it_touches_the_image_border(service, repository):
    service.save(make_jpeg(40, 20), "photo.jpg", [make_box(x=30, y=10, width=10, height=10)])

    assert len(repository.get_dataset().annotations) == 1


def test_should_accept_box_when_it_exceeds_the_border_within_tolerance(service, repository):
    service.save(make_jpeg(40, 20), "photo.jpg", [make_box(x=30, y=10, width=10.005, height=10.005)])

    assert len(repository.get_dataset().annotations) == 1

@pytest.mark.parametrize("box", [
    make_box(x=31, width=10),
    make_box(y=11, height=10),
])
def test_should_raise_when_box_is_outside_the_image(service, box):
    with pytest.raises(InvalidAnnotationError, match=r"Box 0 is outside the image \(40x20\)"):
        service.save(make_jpeg(40, 20), "photo.jpg", [box])


def test_should_report_index_of_the_first_box_outside_the_image(service):
    with pytest.raises(InvalidAnnotationError, match="Box 1 is outside"):
        service.save(make_jpeg(40, 20), "photo.jpg", [make_box(), make_box(x=35)])


def test_should_store_nothing_when_a_box_is_outside_the_image(service, repository, images_dir):
    with pytest.raises(InvalidAnnotationError):
        service.save(make_jpeg(40, 20), "photo.jpg", [make_box(x=35)])

    assert list(images_dir.iterdir()) == []
    assert repository.get_dataset().images == []


def test_should_store_nothing_when_bytes_are_not_an_image(service, repository, images_dir):
    with pytest.raises(InvalidImageError):
        service.save(b"not an image", "photo.jpg", [make_box()])

    assert list(images_dir.iterdir()) == []
    assert repository.get_dataset().images == []

def test_should_check_boxes_against_rotated_size_when_image_has_exif_orientation(service):
    with pytest.raises(InvalidAnnotationError, match=r"\(20x40\)"):
        service.save(make_jpeg(40, 20, orientation=6), "photo.jpg", [make_box(x=25)])


def test_should_store_rotated_file_and_size_when_image_has_exif_orientation(service, repository, images_dir):
    service.save(make_jpeg(40, 20, orientation=6), "photo.jpg", [make_box()])

    image = repository.get_dataset().images[0]
    with Image.open(images_dir / image.file_name) as stored:
        assert stored.size == (20, 40)
        assert stored.getexif().get(EXIF_ORIENTATION_TAG, 1) == 1
    assert (image.width, image.height) == (20, 40)


def test_should_not_overwrite_file_when_same_image_is_saved_again(service, images_dir):
    image_bytes = make_jpeg()
    service.save(image_bytes, "photo.jpg", [make_box()])
    stored = images_dir / f"{hashlib.sha256(image_bytes).hexdigest()}.jpg"
    stored.write_bytes(b"marker")

    service.save(image_bytes, "photo.jpg", [make_box()])

    assert stored.read_bytes() == b"marker"


def test_should_add_new_categories_when_registering_model_categories(service):
    service.register_model_categories({1: "person", 91: "pizza"})

    assert service.get_categories() == ["person", "pizza"]


def test_should_leave_categories_unchanged_when_registering_conflicting_categories(service):
    with pytest.raises(InvalidModelError):
        service.register_model_categories({1: "human", 91: "pizza"})

    assert service.get_categories() == ["person"]


def test_should_delete_record_and_file_when_deleting_images(service, repository, images_dir):
    service.save(make_jpeg(), "photo.jpg", [make_box()])
    image = repository.get_dataset().images[0]

    service.delete_images([image.id])

    assert repository.get_dataset().images == []
    assert not (images_dir / image.file_name).exists()


def test_should_delete_record_when_file_is_already_missing(service, repository, images_dir):
    service.save(make_jpeg(), "photo.jpg", [make_box()])
    image = repository.get_dataset().images[0]
    (images_dir / image.file_name).unlink()

    service.delete_images([image.id])

    assert repository.get_dataset().images == []