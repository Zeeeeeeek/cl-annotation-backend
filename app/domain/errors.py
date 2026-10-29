class InvalidInputError(Exception):
    """Raised when the input data is invalid or cannot be processed."""
    pass

class InvalidImageError(InvalidInputError):
    """Raised when the uploaded image is invalid or cannot be processed."""
    pass

class InvalidAnnotationError(InvalidInputError):
    """Raised when the annotation data is invalid or cannot be processed."""
    pass