import pytest
from pydantic import ValidationError

from app.domain.models import BoundingBox, CategoryInfo, ExpertCorrectedBox, ModelUpdatePayload


def test_should_accept_box_when_origin_is_zero_and_size_is_positive():
    box = BoundingBox(x=0, y=0, width=10, height=5)

    assert (box.x, box.y, box.width, box.height) == (0, 0, 10, 5)


@pytest.mark.parametrize("field", ["x", "y"])
def test_should_reject_box_when_coordinate_is_negative(field):
    values = {"x": 0, "y": 0, "width": 10, "height": 5, field: -1}

    with pytest.raises(ValidationError, match=field):
        BoundingBox(**values)


@pytest.mark.parametrize("field", ["width", "height"])
@pytest.mark.parametrize("size", [0, -1])
def test_should_reject_box_when_size_is_not_positive(field, size):
    values = {"x": 0, "y": 0, "width": 10, "height": 5, field: size}

    with pytest.raises(ValidationError, match=field):
        BoundingBox(**values)


def test_should_strip_and_lowercase_label_when_expert_box_is_created():
    corrected = ExpertCorrectedBox(box=BoundingBox(x=0, y=0, width=1, height=1), label="  Pizza ")

    assert corrected.label == "pizza"


@pytest.mark.parametrize("label", ["", "   "])
def test_should_reject_expert_box_when_label_is_empty_or_blank(label):
    with pytest.raises(ValidationError, match="label"):
        ExpertCorrectedBox(box=BoundingBox(x=0, y=0, width=1, height=1), label=label)


@pytest.mark.parametrize("category_id", [0, -1])
def test_should_reject_category_when_id_is_not_positive(category_id):
    with pytest.raises(ValidationError, match="id"):
        CategoryInfo(id=category_id, name="pizza")


@pytest.mark.parametrize("name", ["", "a" * 101])
def test_should_reject_category_when_name_length_is_out_of_range(name):
    with pytest.raises(ValidationError, match="name"):
        CategoryInfo(id=1, name=name)


def test_should_accept_model_update_when_trained_image_ids_is_empty():
    payload = ModelUpdatePayload(weights_path="w.pt", categories=[CategoryInfo(id=1, name="person")],
                                 trained_image_ids=[])

    assert payload.trained_image_ids == []


def test_should_reject_model_update_when_categories_is_empty():
    with pytest.raises(ValidationError, match="categories"):
        ModelUpdatePayload(weights_path="w.pt", categories=[], trained_image_ids=[])


def test_should_reject_model_update_when_weights_path_is_empty():
    with pytest.raises(ValidationError, match="weights_path"):
        ModelUpdatePayload(weights_path="", categories=[CategoryInfo(id=1, name="person")], trained_image_ids=[])