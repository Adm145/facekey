"""Where enrolled face embeddings get saved.

facekey never needs to keep the original photos — only the embeddings
(the lists of numbers from embedding.py) get stored. One identity can have
several embeddings (one per enrolled photo), since a person doesn't look
identical in every photo.
"""

import os
import sqlite3

import numpy as np

from .errors import UnknownIdentity


class VectorStore:
    """Base class: anything that can save and load named face embeddings.

    facekey ships one implementation (SqliteStore, below). To use your own
    database (Postgres, Qdrant, ...), write a class with these same five
    methods and pass an instance of it to FaceKey(store=...).
    """

    def add(self, name, vectors):
        """Save one or more embeddings under `name` (a list of 1-D numpy
        arrays). Should not remove any embeddings already saved for `name`
        — use this both for a brand-new identity and for adding samples to
        an existing one."""
        raise NotImplementedError

    def get(self, name):
        """Return every embedding saved for `name`, as a list of 1-D numpy
        arrays. Return an empty list if `name` isn't known."""
        raise NotImplementedError

    def all(self):
        """Yield (name, vector) for every single embedding of every
        identity — used by identify() to search everyone at once."""
        raise NotImplementedError

    def delete(self, name):
        """Remove an identity and every embedding it has."""
        raise NotImplementedError

    def names(self):
        """Return a list of every enrolled identity's name."""
        raise NotImplementedError


class SqliteStore(VectorStore):
    """The default store: one SQLite file, using only Python's built-in
    sqlite3 module (no extra dependency to install). Comfortable for a
    personal project or up to a few thousand enrolled identities — past
    that, a real vector database would search faster.
    """

    def __init__(self, path="~/.facekey/faces.db"):
        self.path = os.path.expanduser(path)
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        self._connection = sqlite3.connect(self.path)
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS vectors (
                name   TEXT NOT NULL,
                vector BLOB NOT NULL
            )
            """
        )
        self._connection.commit()

    def add(self, name, vectors):
        rows = [(name, _vector_to_blob(vector)) for vector in vectors]
        self._connection.executemany("INSERT INTO vectors (name, vector) VALUES (?, ?)", rows)
        self._connection.commit()

    def get(self, name):
        cursor = self._connection.execute("SELECT vector FROM vectors WHERE name = ?", (name,))
        return [_blob_to_vector(row[0]) for row in cursor.fetchall()]

    def all(self):
        cursor = self._connection.execute("SELECT name, vector FROM vectors")
        for name, blob in cursor.fetchall():
            yield name, _blob_to_vector(blob)

    def delete(self, name):
        if name not in self.names():
            raise UnknownIdentity(f"No identity named {name!r} is enrolled.")
        self._connection.execute("DELETE FROM vectors WHERE name = ?", (name,))
        self._connection.commit()

    def names(self):
        cursor = self._connection.execute("SELECT DISTINCT name FROM vectors")
        return [row[0] for row in cursor.fetchall()]

    def close(self):
        self._connection.close()


def _vector_to_blob(vector):
    """Turn a numpy array into raw bytes, for storing in a BLOB column."""
    return np.asarray(vector, dtype=np.float32).tobytes()


def _blob_to_vector(blob):
    """The reverse of _vector_to_blob: raw bytes back into a numpy array."""
    return np.frombuffer(blob, dtype=np.float32)
