"""Round-trip tests for SqliteStore. These don't touch the embedding model
or the network at all, so they're safe to run anywhere, anytime.
"""

import numpy as np

from facekey.storage import SqliteStore


def test_add_and_get(tmp_path):
    store = SqliteStore(tmp_path / "faces.db")

    vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    store.add("adam", [vector])

    [saved] = store.get("adam")
    assert np.array_equal(saved, vector)


def test_get_unknown_name_returns_empty_list(tmp_path):
    store = SqliteStore(tmp_path / "faces.db")
    assert store.get("nobody") == []


def test_add_appends_instead_of_replacing(tmp_path):
    store = SqliteStore(tmp_path / "faces.db")

    store.add("adam", [np.array([1.0, 0.0], dtype=np.float32)])
    store.add("adam", [np.array([0.0, 1.0], dtype=np.float32)])

    assert len(store.get("adam")) == 2


def test_names(tmp_path):
    store = SqliteStore(tmp_path / "faces.db")
    store.add("adam", [np.array([1.0], dtype=np.float32)])
    store.add("sam", [np.array([2.0], dtype=np.float32)])

    assert sorted(store.names()) == ["adam", "sam"]


def test_delete_removes_all_of_that_identitys_vectors(tmp_path):
    store = SqliteStore(tmp_path / "faces.db")
    store.add("adam", [np.array([1.0], dtype=np.float32), np.array([2.0], dtype=np.float32)])

    store.delete("adam")

    assert store.get("adam") == []
    assert "adam" not in store.names()


def test_delete_unknown_name_raises(tmp_path):
    import pytest

    from facekey.errors import UnknownIdentity

    store = SqliteStore(tmp_path / "faces.db")
    with pytest.raises(UnknownIdentity):
        store.delete("nobody")


def test_all_yields_every_name_and_vector(tmp_path):
    store = SqliteStore(tmp_path / "faces.db")
    store.add("adam", [np.array([1.0], dtype=np.float32)])
    store.add("sam", [np.array([2.0], dtype=np.float32)])

    pairs = {(name, float(vector[0])) for name, vector in store.all()}
    assert pairs == {("adam", 1.0), ("sam", 2.0)}
