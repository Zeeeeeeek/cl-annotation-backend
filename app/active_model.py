from dataclasses import dataclass
from pathlib import Path

import json


@dataclass
class ActiveModel:
    weights_path: str
    categories: dict[int, str]


def load_active_model_from_json(json_path: str) -> ActiveModel | None:
    file = Path(json_path)
    if not file.exists():
        return None
    data = json.loads(file.read_text())
    weights_path = data["weights_path"]
    if not Path(weights_path).exists():
        raise FileNotFoundError(f"Active model weights file not found: {weights_path}")
    categories = {category["id"]: category["name"] for category in data["categories"]}
    return ActiveModel(weights_path=weights_path, categories=categories)


def save_active_model_to_json(json_path: str, active_model: ActiveModel) -> None:
    data = {
        "weights_path": active_model.weights_path,
        "categories": [{"id": category_id, "name": name} for category_id, name in active_model.categories.items()],
    }
    Path(json_path).write_text(json.dumps(data, indent=2))
