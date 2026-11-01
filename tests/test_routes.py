import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.api import routes
from app.api.web_app import create_web_app
from app.database import create_db_engine
from app.domain.errors import InvalidModelError
from app.domain.models import DetectedBox, BoundingBox
from app.images import open_image
from app.repositories.annotation_repository import AnnotationRepository
from app.services.annotation_service import AnnotationService
from app.services.detection_service import DetectionService

SECRET = "test-secret"


class FakeDetector:
    def __init__(self, active_model=None):
        self.weights_path = active_model.weights_path if active_model else None
        self.categories = active_model.categories if active_model else {1: "person"}

    @staticmethod
    def detect(image_bytes):
        open_image(image_bytes)
        return [DetectedBox(box=BoundingBox(x=1, y=2, width=3, height=4), label="person", confidence=0.9)]

    def get_categories(self):
        return list(self.categories.values())


class FailingDetector:
    def __init__(self, active_model):
        raise InvalidModelError(f"Cannot load weights from {active_model.weights_path}")


def make_jpeg(width=40, height=20):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height)).save(buffer, "JPEG")
    return buffer.getvalue()


def annotations_json(*boxes):
    return json.dumps({"detections": [
        {"box": {"x": x, "y": y, "width": width, "height": height}, "label": label}
        for x, y, width, height, label in boxes
    ]})


def model_payload(categories=None, trained_image_ids=None):
    return {
        "weights_path": "new.pt",
        "categories": categories or [{"id": 1, "name": "person"}, {"id": 91, "name": "pizza"}],
        "trained_image_ids": trained_image_ids or [],
    }


@pytest.fixture
def repository(tmp_path):
    return AnnotationRepository(create_db_engine(str(tmp_path / "test.db")))


@pytest.fixture
def annotation_service(repository, tmp_path):
    return AnnotationService(repository, initial_categories={1: "person"}, images_dir=str(tmp_path / "images"))


@pytest.fixture
def detection_service():
    return DetectionService(FakeDetector())


@pytest.fixture
def active_model_path(tmp_path, monkeypatch):
    path = tmp_path / "active_model.json"
    monkeypatch.setattr(routes, "ACTIVE_MODEL_PATH", str(path))
    monkeypatch.setattr(routes, "MODEL_HOT_UPDATE_SECRET", SECRET)
    monkeypatch.setattr(routes, "create_detector", FakeDetector)
    return path


@pytest.fixture
def client(annotation_service, detection_service):
    return TestClient(create_web_app(detection_service, annotation_service))


def post_annotations(client, image_bytes, annotations):
    return client.post("/annotations", files={"image": ("photo.jpg", image_bytes, "image/jpeg")},
                       data={"annotations": annotations})


def test_should_return_200_when_calling_health(client):
    assert client.get("/health").status_code == 200


def test_should_return_detected_boxes_when_image_is_valid(client):
    response = client.post("/detect", files={"image": ("photo.jpg", make_jpeg(), "image/jpeg")})

    assert response.status_code == 200
    assert response.json() == [
        {"box": {"x": 1, "y": 2, "width": 3, "height": 4}, "label": "person", "confidence": 0.9}
    ]


def test_should_return_422_when_detecting_a_file_that_is_not_an_image(client):
    response = client.post("/detect", files={"image": ("photo.jpg", b"not an image", "image/jpeg")})

    assert response.status_code == 422
    assert response.json() == {"detail": "File is not a supported image"}


def test_should_return_422_when_detect_is_called_without_image(client):
    assert client.post("/detect").status_code == 422


def test_should_save_annotations_when_image_and_boxes_are_valid(client, repository):
    response = post_annotations(client, make_jpeg(), annotations_json((0, 0, 10, 10, "person")))

    assert response.status_code == 200
    assert response.json() == {"status": "success"}
    assert len(repository.get_dataset().annotations) == 1


def test_should_return_422_when_annotations_field_is_missing(client):
    response = client.post("/annotations", files={"image": ("photo.jpg", make_jpeg(), "image/jpeg")})

    assert response.status_code == 422


def test_should_return_422_when_annotations_are_not_valid_json(client, repository):
    response = post_annotations(client, make_jpeg(), "not json")

    assert response.status_code == 422
    assert repository.get_dataset().images == []


def test_should_return_422_with_field_location_when_box_fails_validation(client, repository):
    response = post_annotations(client, make_jpeg(), annotations_json((-1, 0, 10, 10, "person")))

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["detections", 0, "box", "x"]
    assert repository.get_dataset().images == []


def test_should_return_422_when_label_is_blank(client):
    response = post_annotations(client, make_jpeg(), annotations_json((0, 0, 10, 10, "   ")))

    assert response.status_code == 422


def test_should_return_422_when_box_is_outside_the_image(client, repository):
    response = post_annotations(client, make_jpeg(40, 20), annotations_json((35, 0, 10, 10, "person")))

    assert response.status_code == 422
    assert response.json() == {"detail": "Box 0 is outside the image (40x20)"}
    assert repository.get_dataset().images == []


def test_should_return_422_when_annotated_file_is_not_an_image(client):
    response = post_annotations(client, b"not an image", annotations_json())

    assert response.status_code == 422
    assert response.json() == {"detail": "File is not a supported image"}


def test_should_mark_only_model_categories_as_learned_when_listing_classes(client):
    post_annotations(client, make_jpeg(), annotations_json((0, 0, 10, 10, "giraffe")))

    response = client.get("/classes")

    assert response.json() == [{"name": "person", "learned": True}, {"name": "giraffe", "learned": False}]


@pytest.mark.parametrize("headers", [{}, {"X-Model-Secret": "wrong"}])
def test_should_return_403_when_model_secret_is_missing_or_wrong(client, active_model_path, headers):
    response = client.put("/model", json=model_payload(), headers=headers)

    assert response.status_code == 403
    assert not active_model_path.exists()


def test_should_return_403_when_model_secret_is_not_configured(client, active_model_path, monkeypatch):
    monkeypatch.setattr(routes, "MODEL_HOT_UPDATE_SECRET", None)

    response = client.put("/model", json=model_payload(), headers={"X-Model-Secret": SECRET})

    assert response.status_code == 403


def test_should_replace_detector_and_save_active_model_when_update_is_valid(
        client, active_model_path, detection_service, annotation_service):
    response = client.put("/model", json=model_payload(), headers={"X-Model-Secret": SECRET})

    assert response.status_code == 200
    assert detection_service.detector.weights_path == "new.pt"
    assert annotation_service.get_categories() == ["person", "pizza"]
    assert json.loads(active_model_path.read_text()) == {
        "weights_path": "new.pt",
        "categories": [{"id": 1, "name": "person"}, {"id": 91, "name": "pizza"}],
    }


def test_should_delete_trained_images_when_update_is_valid(client, active_model_path, repository):
    post_annotations(client, make_jpeg(), annotations_json((0, 0, 10, 10, "person")))
    image_id = repository.get_dataset().images[0].id

    client.put("/model", json=model_payload(trained_image_ids=[image_id]), headers={"X-Model-Secret": SECRET})

    assert repository.get_dataset().images == []


def test_should_change_nothing_when_model_categories_conflict_with_database(
        client, active_model_path, detection_service):
    response = client.put("/model", json=model_payload(categories=[{"id": 1, "name": "human"}]),
                          headers={"X-Model-Secret": SECRET})

    assert response.status_code == 422
    assert detection_service.detector.weights_path is None
    assert not active_model_path.exists()


def test_should_change_nothing_when_weights_cannot_be_loaded(
        client, active_model_path, detection_service, annotation_service, monkeypatch):
    monkeypatch.setattr(routes, "create_detector", FailingDetector)

    response = client.put("/model", json=model_payload(), headers={"X-Model-Secret": SECRET})

    assert response.status_code == 422
    assert response.json() == {"detail": "Cannot load weights from new.pt"}
    assert detection_service.detector.weights_path is None
    assert annotation_service.get_categories() == ["person"]
    assert not active_model_path.exists()


def test_should_return_422_when_model_update_has_no_categories(client, active_model_path):
    response = client.put("/model", json=model_payload() | {"categories": []}, headers={"X-Model-Secret": SECRET})

    assert response.status_code == 422
