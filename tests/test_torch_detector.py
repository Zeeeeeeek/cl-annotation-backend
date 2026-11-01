import io

import pytest
import torch
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

from app.detectors.TorchDetector import TorchDetector, _to_index_list
from app.domain.errors import InvalidImageError, InvalidModelError

CATEGORIES = {1: "person", 91: "pizza"}


def make_jpeg(width=64, height=48):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height)).save(buffer, "JPEG")
    return buffer.getvalue()


def save_random_weights(path, num_classes):
    model = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, num_classes=num_classes)
    torch.save(model.state_dict(), path)
    return str(path)


class FakeModel:
    def __init__(self, boxes, labels, scores):
        self._prediction = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "scores": torch.tensor(scores, dtype=torch.float32),
        }

    def __call__(self, _images):
        return [self._prediction]


@pytest.fixture(scope="module")
def weights_path(tmp_path_factory):
    return save_random_weights(tmp_path_factory.mktemp("weights") / "custom.pt", num_classes=92)


@pytest.fixture(scope="module")
def custom_detector(weights_path):
    return TorchDetector(weights_path=weights_path, categories=CATEGORIES, confidence_threshold=0.5)


def test_should_put_names_at_their_ids_and_fill_gaps_when_building_index_list():
    assert _to_index_list({1: "person", 3: "car"}) == ["__background__", "person", "N/A", "car"]


def test_should_raise_when_custom_weights_are_given_without_categories():
    with pytest.raises(ValueError, match="categories are required"):
        TorchDetector(weights_path="custom.pt")


def test_should_raise_invalid_model_error_when_weights_file_does_not_exist(tmp_path):
    with pytest.raises(InvalidModelError, match="missing.pt"):
        TorchDetector(weights_path=str(tmp_path / "missing.pt"), categories=CATEGORIES)


def test_should_raise_invalid_model_error_when_weights_file_is_not_a_model(tmp_path):
    weights = tmp_path / "broken.pt"
    weights.write_bytes(b"not a model")

    with pytest.raises(InvalidModelError, match="broken.pt"):
        TorchDetector(weights_path=str(weights), categories=CATEGORIES)


def test_should_raise_invalid_model_error_when_weights_have_a_different_number_of_classes(weights_path):
    with pytest.raises(InvalidModelError, match="custom.pt"):
        TorchDetector(weights_path=weights_path, categories={1: "person"})


def test_should_return_category_names_without_placeholders_when_custom_weights_are_loaded(custom_detector):
    assert custom_detector.get_categories() == ["person", "pizza"]


def test_should_return_categories_with_ids_when_custom_weights_are_loaded(custom_detector):
    assert custom_detector.get_categories_and_ids() == CATEGORIES


def test_should_convert_corners_to_x_y_width_height_when_detecting(custom_detector, monkeypatch):
    monkeypatch.setattr(custom_detector, "_model", FakeModel([[10, 20, 40, 60]], [1], [0.9]))

    box = custom_detector.detect(make_jpeg())[0].box

    assert (box.x, box.y, box.width, box.height) == (10, 20, 30, 40)


def test_should_map_label_index_to_category_name_when_detecting(custom_detector, monkeypatch):
    monkeypatch.setattr(custom_detector, "_model", FakeModel([[0, 0, 1, 1]], [91], [0.9]))

    assert custom_detector.detect(make_jpeg())[0].label == "pizza"


def test_should_keep_only_boxes_at_or_above_threshold_when_detecting(custom_detector, monkeypatch):
    fake = FakeModel([[0, 0, 1, 1], [0, 0, 2, 2], [0, 0, 3, 3]], [1, 1, 1], [0.49, 0.5, 0.9])
    monkeypatch.setattr(custom_detector, "_model", fake)

    confidences = [detected.confidence for detected in custom_detector.detect(make_jpeg())]

    assert confidences == pytest.approx([0.5, 0.9])


def test_should_return_empty_list_when_model_finds_nothing(custom_detector, monkeypatch):
    monkeypatch.setattr(custom_detector, "_model", FakeModel([], [], []))

    assert custom_detector.detect(make_jpeg()) == []


def test_should_raise_invalid_image_error_when_detecting_bytes_that_are_not_an_image(custom_detector):
    with pytest.raises(InvalidImageError):
        custom_detector.detect(b"not an image")


@pytest.mark.slow
def test_should_return_80_coco_categories_when_using_default_weights():
    detector = TorchDetector()

    categories = detector.get_categories_and_ids()

    assert len(categories) == 80
    assert (categories[1], categories[90]) == ("person", "toothbrush")
    assert "N/A" not in categories.values()


@pytest.mark.slow
def test_should_detect_without_errors_when_using_default_weights_on_a_real_forward_pass():
    detections = TorchDetector().detect(make_jpeg())

    assert isinstance(detections, list)
