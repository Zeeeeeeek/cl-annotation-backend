import pytest

from app.database import ImageRecord, create_db_engine
from app.domain.models import BoundingBox, ExpertCorrectedBox
from app.repositories.annotation_repository import AnnotationRepository


@pytest.fixture
def repository(tmp_path):
    return AnnotationRepository(create_db_engine(str(tmp_path / "test.db")))


def make_image(sha256="abc", file_name="abc.jpg"):
    return ImageRecord(sha256=sha256, file_name=file_name, original_name="photo.jpg", width=100, height=50)


def make_box(label, x=0, y=0, width=10, height=10):
    return ExpertCorrectedBox(box=BoundingBox(x=x, y=y, width=width, height=height), label=label)


def test_should_return_empty_dataset_when_database_is_new(repository):
    dataset = repository.get_dataset()

    assert (dataset.images, dataset.annotations, dataset.categories) == ([], [], [])


def test_should_add_categories_when_seeding_an_empty_database(repository):
    repository.seed_categories({1: "person", 91: "pizza"})

    assert repository.get_categories_by_id() == {1: "person", 91: "pizza"}


def test_should_keep_existing_name_when_seeding_an_id_already_in_database(repository):
    repository.seed_categories({1: "person"})

    repository.seed_categories({1: "human", 2: "bicycle"})

    assert repository.get_categories_by_id() == {1: "person", 2: "bicycle"}

def test_should_return_category_names_ordered_by_id(repository):
    repository.seed_categories({91: "pizza", 1: "person", 2: "bicycle"})

    assert repository.get_category_names() == ["person", "bicycle", "pizza"]

def test_should_save_image_and_annotations_when_image_is_new(repository):
    repository.replace_image_annotations(make_image(), [make_box("person", x=1, y=2, width=3, height=4)])

    dataset = repository.get_dataset()
    assert [image.sha256 for image in dataset.images] == ["abc"]
    assert [(a.image_id, a.x, a.y, a.width, a.height) for a in dataset.annotations] == [
        (dataset.images[0].id, 1, 2, 3, 4)]

def test_should_reuse_category_when_label_already_exists(repository):
    repository.seed_categories({1: "person"})

    repository.replace_image_annotations(make_image(), [make_box("person")])

    dataset = repository.get_dataset()
    assert [a.category_id for a in dataset.annotations] == [1]
    assert len(dataset.categories) == 1


def test_should_create_category_once_when_label_is_repeated_in_the_same_image(repository):
    repository.replace_image_annotations(make_image(), [make_box("giraffe"), make_box("giraffe", x=20)])

    dataset = repository.get_dataset()
    assert len(dataset.categories) == 1
    assert len(dataset.annotations) == 2


def test_should_replace_annotations_when_image_with_same_hash_is_saved_again(repository):
    repository.replace_image_annotations(make_image(), [make_box("person"), make_box("pizza")])

    repository.replace_image_annotations(make_image(), [make_box("bicycle", x=5)])

    dataset = repository.get_dataset()
    assert len(dataset.images) == 1
    assert [a.x for a in dataset.annotations] == [5]


def test_should_update_timestamp_when_image_with_same_hash_is_saved_again(repository):
    repository.replace_image_annotations(make_image(), [make_box("person")])

    repository.replace_image_annotations(make_image(), [make_box("person")])

    image = repository.get_dataset().images[0]
    assert image.updated_at > image.created_at


def test_should_keep_image_without_annotations_when_saved_again_with_no_boxes(repository):
    repository.replace_image_annotations(make_image(), [make_box("person")])

    repository.replace_image_annotations(make_image(), [])

    dataset = repository.get_dataset()
    assert len(dataset.images) == 1
    assert dataset.annotations == []


def test_should_return_file_names_and_delete_annotations_when_deleting_images(repository):
    repository.replace_image_annotations(make_image("a", "a.jpg"), [make_box("person")])
    repository.replace_image_annotations(make_image("b", "b.jpg"), [make_box("person")])
    ids = {image.sha256: image.id for image in repository.get_dataset().images}

    deleted = repository.delete_images([ids["a"]])

    dataset = repository.get_dataset()
    assert deleted == ["a.jpg"]
    assert [image.sha256 for image in dataset.images] == ["b"]
    assert [a.image_id for a in dataset.annotations] == [ids["b"]]


def test_should_keep_categories_when_deleting_images(repository):
    repository.replace_image_annotations(make_image(), [make_box("giraffe")])
    image_id = repository.get_dataset().images[0].id

    repository.delete_images([image_id])

    assert list(repository.get_categories_by_id().values()) == ["giraffe"]

def test_should_ignore_unknown_ids_when_deleting_images(repository):
    repository.replace_image_annotations(make_image(), [make_box("person")])

    deleted = repository.delete_images([999])

    assert deleted == []
    assert len(repository.get_dataset().images) == 1