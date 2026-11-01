import json

import pytest

from app.active_model import ActiveModel, load_active_model_from_json, save_active_model_to_json


def test_should_return_none_when_file_does_not_exist(tmp_path):
    assert load_active_model_from_json(str(tmp_path / "active_model.json")) is None


def test_should_load_the_same_model_when_saved_and_loaded_back(tmp_path):
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"")
    model = ActiveModel(weights_path=str(weights), categories={1: "person", 91: "pizza"})
    json_path = str(tmp_path / "active_model.json")

    save_active_model_to_json(json_path, model)

    assert load_active_model_from_json(json_path) == model


def test_should_save_categories_in_coco_format(tmp_path):
    json_path = tmp_path / "active_model.json"

    save_active_model_to_json(str(json_path), ActiveModel(weights_path="w.pt", categories={91: "pizza"}))

    assert json.loads(json_path.read_text())["categories"] == [{"id": 91, "name": "pizza"}]


def test_should_raise_when_weights_file_does_not_exist(tmp_path):
    json_path = tmp_path / "active_model.json"
    json_path.write_text(json.dumps({"weights_path": str(tmp_path / "missing.pt"), "categories": []}))

    with pytest.raises(FileNotFoundError, match="missing.pt"):
        load_active_model_from_json(str(json_path))


def test_should_raise_when_weights_path_is_missing_in_file(tmp_path):
    json_path = tmp_path / "active_model.json"
    json_path.write_text(json.dumps({"categories": []}))

    with pytest.raises(KeyError, match="weights_path"):
        load_active_model_from_json(str(json_path))