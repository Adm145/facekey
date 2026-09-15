"""Liveness checking: telling a real face apart from a photo, screen, or
video held up to the camera.

facekey doesn't ship a working liveness checker yet — that's planned as a
separate, optional facekey[liveness] extra. This file only defines the
shape a liveness checker needs to have, so you can write your own (or the
built-in one, once it exists) and plug it into FaceKey the same way.
"""


class LivenessChecker:
    """Base class for a liveness/anti-spoof check.

    Write a subclass with a check() method and pass an instance of it to
    FaceKey(liveness=...). Leaving liveness unset (the default) means
    facekey does no spoof-resistance at all — verify()/identify() only
    check "is this the same face", not "is this a real, live face" — which
    is fine for a low-stakes convenience feature, but not something to rely
    on for anything security-sensitive on its own.
    """

    def check(self, images):
        """Given one or more images (the same forgiving formats embed()
        accepts — file path, bytes, or numpy array), return True if they
        look like a live face and False if they look like a spoof attempt.
        """
        raise NotImplementedError
