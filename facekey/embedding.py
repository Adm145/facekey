"""Turning a photo into a face embedding.

An "embedding" here is just a list of 512 numbers that represents a face —
two photos of the same person produce two similar lists of numbers, two
photos of different people produce dissimilar ones. That's the whole trick
behind register()/verify()/identify() in core.py.
"""

import cv2
import numpy as np

from .errors import MultipleFacesDetected, NoFaceDetected


class Embedder:
    """Base class: anything that can turn a face photo into an embedding.

    facekey ships one implementation (InsightFaceEmbedder, below). To use a
    different model, write your own class with the same embed() method and
    pass an instance of it to FaceKey(embedder=...).
    """

    # Subclasses should set this to the length of the vector embed() returns.
    dim = None

    def embed(self, image):
        """Return one embedding (a 1-D numpy array of length `dim`) for the
        single face found in `image`.

        Should raise NoFaceDetected if there's no face, or
        MultipleFacesDetected if there's more than one and that isn't
        allowed.
        """
        raise NotImplementedError


class InsightFaceEmbedder(Embedder):
    """The default embedder, backed by InsightFace's `buffalo_l` model pack.

    Only the "detection" and "recognition" pieces of that pack are loaded —
    buffalo_l also ships age/gender and 3D-landmark models that facekey has
    no use for, so they're never loaded into memory.

    Note on licensing: InsightFace's pretrained recognition weights are
    released for non-commercial research use. Using InsightFaceEmbedder in
    a commercial product means either negotiating a license for those
    weights or swapping in a different Embedder before shipping. See the
    project README for details.

    Nothing here touches the network until embed() (or preload()) is
    actually called for the first time — creating this object is instant.
    """

    dim = 512

    def __init__(self, model_dir=None, multi_face="reject"):
        """
        model_dir: where InsightFace should look for (and download, if
            missing) its model files. Defaults to InsightFace's own default
            location (~/.insightface).
        multi_face: what to do if a photo has more than one face in it.
            "reject" (default) raises MultipleFacesDetected. "largest" picks
            the biggest face in the photo and ignores the rest.
        """
        if multi_face not in ("reject", "largest"):
            raise ValueError('multi_face must be "reject" or "largest"')

        self.model_dir = model_dir
        self.multi_face = multi_face
        self._app = None  # the InsightFace model; created on first use

    def preload(self):
        """Load the model right now, instead of waiting for the first
        embed() call. Useful to call once at startup so the (one-time,
        ~275MB) download and load don't happen in the middle of handling a
        user's first request."""
        self._ensure_loaded()

    def _ensure_loaded(self):
        if self._app is not None:
            return

        # Imported here, not at the top of the file, so simply importing
        # facekey doesn't pull in onnxruntime/insightface until something
        # actually needs the model.
        from insightface.app import FaceAnalysis

        kwargs = {"allowed_modules": ["detection", "recognition"]}
        if self.model_dir is not None:
            kwargs["root"] = self.model_dir

        app = FaceAnalysis(name="buffalo_l", **kwargs)
        app.prepare(ctx_id=-1)  # ctx_id=-1 means "run on CPU"
        self._app = app

    def embed(self, image):
        self._ensure_loaded()

        image_bgr = _load_as_bgr(image)
        faces = self._app.get(image_bgr)

        if not faces:
            raise NoFaceDetected("No face found in the given image.")

        if len(faces) > 1:
            if self.multi_face == "reject":
                raise MultipleFacesDetected(
                    f"Found {len(faces)} faces in one image; expected exactly one."
                )
            faces = sorted(faces, key=_face_area, reverse=True)

        return _l2_normalize(faces[0].embedding)


def _face_area(face):
    """Width * height of a detected face's bounding box, used to find the
    largest face when multi_face="largest"."""
    x1, y1, x2, y2 = face.bbox
    return (x2 - x1) * (y2 - y1)


def _l2_normalize(vector):
    """Scale a vector so its length is exactly 1.

    Doing this once here means comparing two embeddings later is a single
    dot product (cosine similarity), instead of a more expensive formula
    every time.
    """
    vector = np.asarray(vector, dtype=np.float32)
    length = np.linalg.norm(vector)
    if length == 0:
        return vector
    return vector / length


def _load_as_bgr(image):
    """Accept a file path, raw image bytes, or an already-decoded numpy
    array, and return a decoded image in OpenCV's BGR format."""

    if isinstance(image, np.ndarray):
        return image

    if isinstance(image, (bytes, bytearray)):
        array = np.frombuffer(image, dtype=np.uint8)
        decoded = cv2.imdecode(array, cv2.IMREAD_COLOR)
    elif isinstance(image, str):
        decoded = cv2.imread(image)
    else:
        raise TypeError(
            f"Don't know how to read an image from a {type(image).__name__!r}; "
            "pass a file path, bytes, or a numpy array."
        )

    if decoded is None:
        raise NoFaceDetected("Could not read the given image (it may be corrupt or an unsupported format).")

    return decoded
