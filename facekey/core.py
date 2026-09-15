"""FaceKey: the main class most people using this library will interact
with. It ties together an Embedder (turns a photo into a face embedding),
a VectorStore (saves/loads embeddings by name), and optionally a
LivenessChecker (tells a real face apart from a photo of a photo).
"""

import logging

import numpy as np

from .embedding import InsightFaceEmbedder
from .errors import IdentityAlreadyExists, UnknownIdentity
from .storage import SqliteStore
from .types import IdentifyResult, VerifyResult

logger = logging.getLogger(__name__)


class FaceKey:
    def __init__(self, store="~/.facekey/faces.db", liveness=None, embedder=None, threshold=0.42):
        """
        store: where enrolled faces are saved. Either a file path (a
            SqliteStore is created there automatically), or your own
            VectorStore instance.
        liveness: a LivenessChecker instance, or None (the default) to skip
            liveness checking entirely. See liveness.py — facekey doesn't
            ship a built-in checker yet.
        embedder: an Embedder instance, or None (the default) to use
            InsightFaceEmbedder.
        threshold: how similar two embeddings need to be (cosine
            similarity, 0.0-1.0) to count as the same person. 0.42 is a
            reasonable starting point for the default model; tune it for
            your own use case.
        """
        self.store = SqliteStore(store) if isinstance(store, str) else store
        self.embedder = embedder if embedder is not None else InsightFaceEmbedder()
        self.liveness = liveness
        self.threshold = threshold

        if liveness is None:
            logger.warning(
                "FaceKey was created without a liveness checker - verify()/identify() only "
                "check whether a face matches, not whether it's a real live face. See "
                "facekey.liveness for details."
            )

    def preload(self):
        """Load the embedding model right now, instead of on first use.
        Call this once at startup if you don't want the first real
        register()/verify()/identify() call to pay for it."""
        if hasattr(self.embedder, "preload"):
            self.embedder.preload()

    def list(self):
        """Return the names of everyone currently enrolled."""
        return self.store.names()

    def register(self, name, images):
        """Enroll a brand-new identity from one or more photos.

        Raises IdentityAlreadyExists if `name` is already enrolled — call
        add_samples() instead if you want to add more photos to them.
        """
        if name in self.store.names():
            raise IdentityAlreadyExists(f"{name!r} is already enrolled; use add_samples() to add more photos.")

        vectors = [self.embedder.embed(image) for image in _as_list(images)]
        self.store.add(name, vectors)

    def add_samples(self, name, images):
        """Add more reference photos to an identity that's already
        enrolled. Raises UnknownIdentity if `name` hasn't been registered."""
        if name not in self.store.names():
            raise UnknownIdentity(f"No identity named {name!r} is enrolled; use register() first.")

        vectors = [self.embedder.embed(image) for image in _as_list(images)]
        self.store.add(name, vectors)

    def delete(self, name):
        """Remove an identity and every photo they were enrolled with."""
        if name not in self.store.names():
            raise UnknownIdentity(f"No identity named {name!r} is enrolled.")
        self.store.delete(name)

    def verify(self, name, images):
        """1:1 check: does `images` look like the named identity?

        Compares against every reference photo `name` was enrolled with
        and keeps the best (highest) score, since a person doesn't look
        identical in every photo.
        """
        if name not in self.store.names():
            raise UnknownIdentity(f"No identity named {name!r} is enrolled.")

        images = _as_list(images)
        if self.liveness is not None and not self.liveness.check(images):
            return VerifyResult(match=False, score=0.0)

        reference_vectors = self.store.get(name)
        query_vectors = [self.embedder.embed(image) for image in images]

        score = max(_cosine_similarity(query, reference) for query in query_vectors for reference in reference_vectors)
        return VerifyResult(match=score >= self.threshold, score=score)

    def identify(self, images, top_k=1):
        """1:N check: who (if anyone) does `images` match, out of everyone
        enrolled? Returns the best-scoring identities whose score clears
        `threshold`, best first — an empty list means no one matched."""
        images = _as_list(images)
        if self.liveness is not None and not self.liveness.check(images):
            return []

        query_vectors = [self.embedder.embed(image) for image in images]

        best_score_by_name = {}
        for enrolled_name, reference_vector in self.store.all():
            score = max(_cosine_similarity(query, reference_vector) for query in query_vectors)
            if score > best_score_by_name.get(enrolled_name, -1.0):
                best_score_by_name[enrolled_name] = score

        candidates = [
            IdentifyResult(name=name, score=score)
            for name, score in best_score_by_name.items()
            if score >= self.threshold
        ]
        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        return candidates[:top_k]


def _cosine_similarity(a, b):
    """How similar two embeddings are, from -1 (opposite) to 1 (identical).
    Since embed() always returns unit-length vectors, this is just a dot
    product — no division needed."""
    return float(np.dot(a, b))


def _as_list(images):
    """Let callers pass either one image or a list of images."""
    if isinstance(images, (list, tuple)):
        return list(images)
    return [images]
