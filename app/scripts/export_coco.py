"""Export every image, annotation and category in the database to one COCO JSON file.

  Usage (from the backend folder, with the same env vars as the backend):
      python -m app.scripts.export_coco OUTPUT_FILE

  Example:
      python -m app.scripts.export_coco data/export.json

  What it reads:
      DATABASE_PATH  SQLite database (default data/annotations.db)
      IMAGES_DIR     folder with the stored images (default data/images)

  What it writes:
      OUTPUT_FILE, overwritten without asking (its folder must exist), with:
      - images: one entry per stored image, with
          id         database id of the image; annotations point to it with
                     image_id, and PUT /model takes these ids in
                     trained_image_ids to delete the images after training
          file_name  name of the stored file inside info.image_dir, NOT the
                     name the user uploaded: sha256 of the uploaded bytes plus
                     the original extension in lowercase, e.g. "3fa9...c2.jpg".
                     Full path = info.image_dir / file_name
          width, height  size in pixels after EXIF rotation (the stored file
                     is already rotated, so it matches)
      - annotations: id, image_id, category_id, bbox [x, y, width, height] in
        pixels, area, iscrowd 0
      - categories: all of them, also those with no annotations
      - info.image_dir: absolute path of IMAGES_DIR, in the format of the OS the
        script runs on (run it in WSL if the training service runs there)

  Images are not copied and nothing in the database is changed. If the database
  does not exist, an empty one is created and the export is empty.
"""

import json
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path

from app.config import DATABASE_PATH, IMAGES_DIR
from app.repositories.database import create_db_engine
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
    parser = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("output_file", help="path of the COCO JSON file to write (overwritten if it exists)")
    args = parser.parse_args()

    repository = AnnotationRepository(create_db_engine(DATABASE_PATH))
    coco = to_coco(repository.get_dataset())
    coco["info"] = {"image_dir": str(Path(IMAGES_DIR).resolve())}

    Path(args.output_file).write_text(json.dumps(coco, indent=2))
    print(f"Exported {len(coco['images'])} images and {len(coco['annotations'])} annotations to {args.output_file}")


if __name__ == "__main__":
    main()
