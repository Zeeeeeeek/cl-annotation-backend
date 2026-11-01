import io

import pytest
from PIL import Image

from app.domain.errors import InvalidImageError
from app.domain.images import open_image

EXIF_ORIENTATION_TAG = 274


def make_jpeg(width, height, orientation=None):
    exif = Image.Exif()
    if orientation is not None:
        exif[EXIF_ORIENTATION_TAG] = orientation
    buffer = io.BytesIO()
    Image.new("RGB", (width, height)).save(buffer, "JPEG", exif=exif)
    return buffer.getvalue()


def test_should_return_image_and_format_when_bytes_are_a_valid_jpeg():
    opened = open_image(make_jpeg(40, 20))

    assert opened.format == "JPEG"
    assert opened.image.size == (40, 20)


def test_should_not_mark_as_rotated_when_exif_orientation_is_missing():
    opened = open_image(make_jpeg(40, 20))

    assert opened.was_rotated is False


def test_should_not_mark_as_rotated_when_exif_orientation_is_1():
    opened = open_image(make_jpeg(40, 20, orientation=1))

    assert opened.was_rotated is False
    assert opened.image.size == (40, 20)


def test_should_rotate_and_mark_as_rotated_when_exif_orientation_is_6():
    opened = open_image(make_jpeg(40, 20, orientation=6))

    assert opened.was_rotated is True
    assert opened.image.size == (20, 40)


def test_should_raise_invalid_image_error_when_bytes_are_not_an_image():
    with pytest.raises(InvalidImageError, match="File is not a supported image"):
        open_image(b"not an image")


def test_should_raise_invalid_image_error_when_bytes_are_empty():
    with pytest.raises(InvalidImageError, match="File is not a supported image"):
        open_image(b"")