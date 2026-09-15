"""All the exceptions facekey can raise, in one place.

They all inherit from FaceKeyError, so callers who don't care about the
exact reason can just do `except FaceKeyError:` and catch anything from
this library.
"""


class FaceKeyError(Exception):
    """Base class for every error facekey raises."""


class NoFaceDetected(FaceKeyError):
    """Raised when an image was given but no face could be found in it."""


class MultipleFacesDetected(FaceKeyError):
    """Raised when an image has more than one face and that isn't allowed
    (see the `multi_face` option on InsightFaceEmbedder)."""


class UnknownIdentity(FaceKeyError):
    """Raised when asking about a name that hasn't been registered."""


class IdentityAlreadyExists(FaceKeyError):
    """Raised by register() when the name is already enrolled.

    Use add_samples() instead if you want to add more photos to someone
    who's already registered.
    """
