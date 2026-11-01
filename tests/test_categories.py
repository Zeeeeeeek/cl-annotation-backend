import pytest

from app.domain.errors import InvalidModelError
from app.services.annotation_service import _check_categories_match


def test_should_accept_when_model_and_database_categories_are_equal():
    _check_categories_match({1: "person", 91: "pizza"}, {1: "person", 91: "pizza"})


def test_should_accept_when_database_is_empty():
    _check_categories_match({1: "person"}, {})


def test_should_accept_when_database_has_categories_unknown_to_the_model():
    _check_categories_match({1: "person"}, {1: "person", 92: "giraffe"})


def test_should_raise_when_id_has_a_different_name_in_database():
    with pytest.raises(InvalidModelError, match="Category id 1 is 'person' in the model but 'human' in the database"):
        _check_categories_match({1: "person"}, {1: "human"})


def test_should_raise_when_name_has_a_different_id_in_database():
    with pytest.raises(InvalidModelError, match="has id 91 in the model but 95 in the database"):
        _check_categories_match({91: "pizza"}, {95: "pizza"})