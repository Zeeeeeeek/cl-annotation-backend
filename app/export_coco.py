import json
from argparse import ArgumentParser
from pathlib import Path

from app.config import DATABASE_PATH, IMAGES_DIR
from app.database import create_db_engine
from app.repositories.annotation_repository import Dataset, AnnotationRepository


def to_coco(dataset: Dataset) -> dict:
    return {
        "images": [
            {
                "id": image.id,
                "file_name": image.file_name,
                "width": image.width,
                "height": image.height,
            }
            for image in dataset.images
        ],
        "annotations": [
            {
                "id": annotation.id,
                "image_id": annotation.image_id,
                "category_id": annotation.category_id,
                "bbox": [annotation.x, annotation.y, annotation.width, annotation.height],
                "area": annotation.width * annotation.height,
                "iscrowd": 0,
            }
            for annotation in dataset.annotations
        ],
        "categories": [{
            "id": category.id,
            "name": category.name,
        }
            for category in dataset.categories
        ],
    }

def main():
    parser = ArgumentParser(description="Export dataset to COCO format")
    parser.add_argument("output_file", help="Path to the output JSON file")
    args = parser.parse_args()

    repository = AnnotationRepository(create_db_engine(DATABASE_PATH))
    coco = to_coco(repository.get_dataset())
    coco["info"] = {"image_dir": str(Path(IMAGES_DIR).resolve())}

    Path(args.output_file).write_text(json.dumps(coco, indent=2))
    print(f"Exported {len(coco['images'])} images and {len(coco['annotations'])} annotations to {args.output_file}")

if __name__ == "__main__":
    main()