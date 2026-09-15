# facekey

A small Python library for adding software face authentication to your own
app — enroll a face, then check a later photo against it. Similar in
concept to Face ID, but running on an ordinary camera with no depth sensor,
so treat it as a convenience factor rather than a standalone security
measure.

Face detection and recognition are done with
[InsightFace](https://github.com/deepinsight/insightface) (only its
detection + recognition models — none of the extra age/gender/landmark
models it also ships are loaded). Enrolled faces are stored locally by
default (SQLite, no server required).

> **Status:** early, in-progress rewrite. The API below is the current
> plan and may still change.

## Install

```bash
pip install -e .
```

## Usage

```python
from facekey import FaceKey

fk = FaceKey()  # stores enrolled faces at ~/.facekey/faces.db by default

fk.register("adam", ["photo1.jpg", "photo2.jpg", "photo3.jpg"])

result = fk.verify("adam", "new_photo.jpg")
print(result.match, result.score)

fk.add_samples("adam", ["another_photo.jpg"])  # add more reference photos later
fk.delete("adam")                              # remove an identity entirely
```

`register`/`verify`/`add_samples`/`identify` all accept a file path, raw
image bytes, or a numpy array — pick whichever's convenient.

## Licensing note

InsightFace's pretrained recognition weights (used here for the embedding
step) are released for **non-commercial research use**. Swap in your own
`Embedder` (see `facekey/embedding.py`) before using this in anything
commercial.

## Design

- `FaceKey` (`facekey/core.py`) — the main class: register, verify,
  identify, add_samples, delete, list, preload.
- `Embedder` (`facekey/embedding.py`) — turns a photo into a face
  embedding. Default: `InsightFaceEmbedder`.
- `VectorStore` (`facekey/storage.py`) — saves/loads embeddings by name.
  Default: `SqliteStore`.
- `LivenessChecker` (`facekey/liveness.py`) — interface only for now
  (no default implementation yet); tells a real face apart from a photo or
  screen held up to the camera.

Each of these is a plain base class you can subclass to plug in your own
model, database, or liveness check.

## Development

```bash
pip install -e ".[dev]"
pytest
```
