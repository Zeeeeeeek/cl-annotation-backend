import io
from dataclasses import dataclass
from PIL import Image, ImageOps

from app.domain.errors import InvalidImageError


@dataclass
class OpenedImage:
    image: Image.Image
    format: str | None
    was_rotated: bool

def open_image(image_bytes: bytes) -> OpenedImage:
    try:
        original = Image.open(io.BytesIO(image_bytes))
        was_rotated = original.getexif().get(274, 1) != 1  # Check if the image has been rotated based on EXIF orientation
        return OpenedImage(image=ImageOps.exif_transpose(original), format=original.format, was_rotated=was_rotated)
    except OSError as error:
        raise InvalidImageError("File is not a supported image") from error