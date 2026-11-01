from app.database import Annotation, Category, ImageRecord
from app.export_coco import to_coco
from app.repositories.annotation_repository import Dataset


def make_dataset():
    return Dataset(
        images=[ImageRecord(id=1, sha256="abc", file_name="abc.jpg", original_name="photo.jpg", width=100, height=50)],
        annotations=[Annotation(id=7, image_id=1, category_id=91, x=10, y=20, width=30, height=40)],
        categories=[Category(id=91, name="pizza")],
    )


def test_should_return_empty_lists_when_dataset_is_empty():
    assert to_coco(Dataset(images=[], annotations=[], categories=[])) == {"images": [], "annotations": [],
                                                                          "categories": []}


def test_should_export_image_with_stored_file_name_and_size():
    assert to_coco(make_dataset())["images"] == [{"id": 1, "file_name": "abc.jpg", "width": 100, "height": 50}]


def test_should_export_annotation_with_bbox_as_x_y_width_height():
    assert to_coco(make_dataset())["annotations"][0]["bbox"] == [10, 20, 30, 40]


def test_should_compute_area_as_width_times_height():
    assert to_coco(make_dataset())["annotations"][0]["area"] == 1200


def test_should_export_annotation_with_ids_and_iscrowd_zero():
    annotation = to_coco(make_dataset())["annotations"][0]

    assert (annotation["id"], annotation["image_id"], annotation["category_id"], annotation["iscrowd"]) == (7, 1, 91, 0)


def test_should_export_categories_with_id_and_name():
    assert to_coco(make_dataset())["categories"] == [{"id": 91, "name": "pizza"}]


def test_should_not_include_info_when_converting_dataset():
    assert "info" not in to_coco(make_dataset())