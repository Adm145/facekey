"""Small plain objects that facekey hands back to you as results.

These use @dataclass, a built-in Python shortcut for "a class that's just a
few named fields". Writing `@dataclass` above a class with fields listed
like below automatically gives you an `__init__` that sets them, and a
readable `__repr__` for printing — without typing that boilerplate by hand.
"""

from dataclasses import dataclass


@dataclass
class VerifyResult:
    """What you get back from FaceKey.verify() — a 1:1 check against one
    named identity."""

    match: bool
    score: float


@dataclass
class IdentifyResult:
    """One candidate from FaceKey.identify() — a 1:N check against everyone
    who's enrolled. score is a cosine similarity: 1.0 is a perfect match,
    0.0 means completely unrelated faces."""

    name: str
    score: float
