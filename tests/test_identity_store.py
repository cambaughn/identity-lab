"""IdentityStore: CRUD, round-trips, corruption, migrations, concurrency.

This layer must be the most reliable part of the project — tests bias
heavily toward edge cases and hostile data. No webcam, no model, no Qt.
"""

import sqlite3
import threading

import numpy as np
import pytest

from identity_lab.identity.errors import (
    CorruptedDatabaseError,
    DatabaseUnavailableError,
    DuplicateIdentityError,
    IdentityNotFoundError,
    MalformedEmbeddingError,
    ModelMismatchError,
    SchemaMismatchError,
)
from identity_lab.identity.store import SCHEMA_VERSION, IdentityStore

MODEL = "buffalo_l"


@pytest.fixture
def store(tmp_path):
    s = IdentityStore(tmp_path / "identities.db")
    yield s
    s.close()


def vec(seed=0, dim=512):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------- identity CRUD

def test_create_and_get_identity(store):
    rec = store.create_identity("Cam")
    assert rec.display_name == "Cam"
    assert rec.sample_count == 0
    assert rec.model_id is None and rec.dim is None
    assert store.get_identity(rec.identity_id) == rec


def test_create_strips_whitespace(store):
    rec = store.create_identity("  Ale  ")
    assert rec.display_name == "Ale"


def test_create_empty_name_rejected(store):
    with pytest.raises(ValueError):
        store.create_identity("   ")


def test_duplicate_name_rejected_case_insensitive(store):
    store.create_identity("Cam")
    with pytest.raises(DuplicateIdentityError):
        store.create_identity("cam")


def test_unicode_names(store):
    rec = store.create_identity("Алёна 王")
    assert store.get_identity(rec.identity_id).display_name == "Алёна 王"


def test_list_identities_ordered_by_creation(store):
    a = store.create_identity("A")
    b = store.create_identity("B")
    ids = [r.identity_id for r in store.list_identities()]
    assert ids == [a.identity_id, b.identity_id]


def test_get_missing_identity_raises(store):
    with pytest.raises(IdentityNotFoundError):
        store.get_identity("nope")


def test_rename_identity(store):
    rec = store.create_identity("Cam")
    renamed = store.rename_identity(rec.identity_id, "Cameron")
    assert renamed.display_name == "Cameron"
    assert store.get_identity(rec.identity_id).display_name == "Cameron"


def test_rename_missing_raises(store):
    with pytest.raises(IdentityNotFoundError):
        store.rename_identity("nope", "X")


def test_rename_to_existing_name_rejected(store):
    store.create_identity("Cam")
    other = store.create_identity("Ale")
    with pytest.raises(DuplicateIdentityError):
        store.rename_identity(other.identity_id, "CAM")


def test_rename_to_same_name_is_allowed(store):
    rec = store.create_identity("Cam")
    assert store.rename_identity(rec.identity_id, "Cam").display_name == "Cam"


def test_delete_identity(store):
    rec = store.create_identity("Cam")
    store.delete_identity(rec.identity_id)
    assert store.list_identities() == []
    with pytest.raises(IdentityNotFoundError):
        store.get_identity(rec.identity_id)


def test_delete_missing_raises(store):
    with pytest.raises(IdentityNotFoundError):
        store.delete_identity("nope")


def test_delete_frees_name_for_reuse(store):
    rec = store.create_identity("Cam")
    store.delete_identity(rec.identity_id)
    assert store.create_identity("Cam").display_name == "Cam"


# ---------------------------------------------------------------- samples

def test_sample_round_trip_byte_exact(store):
    rec = store.create_identity("Cam")
    original = vec(1)
    store.add_embedding_sample(rec.identity_id, original, MODEL)
    [sample] = store.get_embedding_samples(rec.identity_id)
    assert sample.dim == 512
    assert sample.dtype == "float32"
    assert sample.model_id == MODEL
    assert sample.embedding.dtype == np.float32
    assert np.array_equal(sample.embedding, original)  # byte-exact
    assert sample.embedding.flags.writeable


def test_samples_in_insertion_order(store):
    rec = store.create_identity("Cam")
    for i in range(5):
        store.add_embedding_sample(rec.identity_id, vec(i), MODEL)
    samples = store.get_embedding_samples(rec.identity_id)
    assert len(samples) == 5
    for i, s in enumerate(samples):
        assert np.array_equal(s.embedding, vec(i))


def test_sample_count_and_model_reported_on_identity(store):
    rec = store.create_identity("Cam")
    store.add_embedding_sample(rec.identity_id, vec(1), MODEL)
    store.add_embedding_sample(rec.identity_id, vec(2), MODEL)
    updated = store.get_identity(rec.identity_id)
    assert updated.sample_count == 2
    assert updated.model_id == MODEL
    assert updated.dim == 512


def test_add_sample_missing_identity_raises(store):
    with pytest.raises(IdentityNotFoundError):
        store.add_embedding_sample("nope", vec(), MODEL)


def test_get_samples_missing_identity_raises(store):
    with pytest.raises(IdentityNotFoundError):
        store.get_embedding_samples("nope")


def test_delete_identity_cascades_samples(store, tmp_path):
    rec = store.create_identity("Cam")
    store.add_embedding_sample(rec.identity_id, vec(), MODEL)
    store.delete_identity(rec.identity_id)
    with sqlite3.connect(tmp_path / "identities.db") as conn:
        count = conn.execute("SELECT COUNT(*) FROM embedding_samples").fetchone()[0]
    assert count == 0


# ---------------------------------------------------------------- validation

@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((2, 256), dtype=np.float32),        # 2-D
        np.zeros(0, dtype=np.float32),               # empty
        vec().astype(np.float64),                    # wrong dtype
        np.zeros(512, dtype=np.int32),               # integer dtype
        [0.0] * 512,                                 # not an ndarray
    ],
)
def test_malformed_embeddings_rejected_on_write(store, bad):
    rec = store.create_identity("Cam")
    with pytest.raises(MalformedEmbeddingError):
        store.add_embedding_sample(rec.identity_id, bad, MODEL)


def test_nan_and_inf_rejected_on_write(store):
    rec = store.create_identity("Cam")
    for poison in (np.nan, np.inf):
        v = vec()
        v[7] = poison
        with pytest.raises(MalformedEmbeddingError):
            store.add_embedding_sample(rec.identity_id, v, MODEL)


def test_empty_model_id_rejected(store):
    rec = store.create_identity("Cam")
    with pytest.raises(ValueError):
        store.add_embedding_sample(rec.identity_id, vec(), "  ")


def test_mixed_model_rejected(store):
    rec = store.create_identity("Cam")
    store.add_embedding_sample(rec.identity_id, vec(), MODEL)
    with pytest.raises(ModelMismatchError):
        store.add_embedding_sample(rec.identity_id, vec(), "other_model")


def test_mixed_dim_rejected(store):
    rec = store.create_identity("Cam")
    store.add_embedding_sample(rec.identity_id, vec(dim=512), MODEL)
    with pytest.raises(ModelMismatchError):
        store.add_embedding_sample(rec.identity_id, vec(dim=128), MODEL)


def test_different_identities_may_use_different_models(store):
    a = store.create_identity("A")
    b = store.create_identity("B")
    store.add_embedding_sample(a.identity_id, vec(dim=512), MODEL)
    store.add_embedding_sample(b.identity_id, vec(dim=128), "small_model")
    assert store.get_identity(b.identity_id).dim == 128


# ---------------------------------------------------------------- hostile data

def _tamper(db_path, sql, params=()):
    with sqlite3.connect(db_path) as conn:
        conn.execute(sql, params)


def test_truncated_blob_detected_on_read(store, tmp_path):
    rec = store.create_identity("Cam")
    store.add_embedding_sample(rec.identity_id, vec(), MODEL)
    _tamper(tmp_path / "identities.db",
            "UPDATE embedding_samples SET embedding = X'DEADBEEF'")
    with pytest.raises(MalformedEmbeddingError):
        store.get_embedding_samples(rec.identity_id)


def test_dim_mismatch_detected_on_read(store, tmp_path):
    rec = store.create_identity("Cam")
    store.add_embedding_sample(rec.identity_id, vec(), MODEL)
    _tamper(tmp_path / "identities.db",
            "UPDATE embedding_samples SET dim = 256")
    with pytest.raises(MalformedEmbeddingError):
        store.get_embedding_samples(rec.identity_id)


def test_unknown_dtype_detected_on_read(store, tmp_path):
    rec = store.create_identity("Cam")
    store.add_embedding_sample(rec.identity_id, vec(), MODEL)
    _tamper(tmp_path / "identities.db",
            "UPDATE embedding_samples SET dtype = 'float16'")
    with pytest.raises(MalformedEmbeddingError):
        store.get_embedding_samples(rec.identity_id)


def test_nan_in_stored_blob_detected_on_read(store, tmp_path):
    rec = store.create_identity("Cam")
    store.add_embedding_sample(rec.identity_id, vec(), MODEL)
    poisoned = vec()
    poisoned[0] = np.nan
    blob = np.ascontiguousarray(poisoned, dtype="<f4").tobytes()
    _tamper(tmp_path / "identities.db",
            "UPDATE embedding_samples SET embedding = ?", (blob,))
    with pytest.raises(MalformedEmbeddingError):
        store.get_embedding_samples(rec.identity_id)


def test_not_a_database_file(tmp_path):
    path = tmp_path / "garbage.db"
    path.write_bytes(b"this is definitely not sqlite" * 100)
    with pytest.raises(CorruptedDatabaseError):
        IdentityStore(path)


def test_path_is_directory(tmp_path):
    with pytest.raises(DatabaseUnavailableError):
        IdentityStore(tmp_path)  # a directory, not a file


def test_operations_after_close_raise(tmp_path):
    store = IdentityStore(tmp_path / "identities.db")
    store.close()
    with pytest.raises(DatabaseUnavailableError):
        store.list_identities()
    with pytest.raises(DatabaseUnavailableError):
        store.create_identity("Cam")


def test_context_manager_closes(tmp_path):
    with IdentityStore(tmp_path / "identities.db") as store:
        store.create_identity("Cam")
    with pytest.raises(DatabaseUnavailableError):
        store.list_identities()


# ---------------------------------------------------------------- migrations

def test_migration_records_version(store, tmp_path):
    with sqlite3.connect(tmp_path / "identities.db") as conn:
        rows = conn.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
    assert [r[0] for r in rows] == list(range(1, SCHEMA_VERSION + 1))


def test_reopen_is_idempotent_and_preserves_data(tmp_path):
    path = tmp_path / "identities.db"
    with IdentityStore(path) as store:
        rec = store.create_identity("Cam")
        store.add_embedding_sample(rec.identity_id, vec(9), MODEL)
    with IdentityStore(path) as store:  # reopen: migrations must not re-run
        [reloaded] = store.list_identities()
        assert reloaded.display_name == "Cam"
        [sample] = store.get_embedding_samples(reloaded.identity_id)
        assert np.array_equal(sample.embedding, vec(9))
    with sqlite3.connect(path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0]
    assert count == SCHEMA_VERSION


def test_newer_schema_version_refused(tmp_path):
    path = tmp_path / "identities.db"
    IdentityStore(path).close()
    _tamper(path,
            "INSERT INTO schema_migrations (version, applied_at) VALUES (999, 'x')")
    with pytest.raises(SchemaMismatchError):
        IdentityStore(path)


# ---------------------------------------------------------------- reset

def test_reset_database_clears_everything_and_store_remains_usable(store):
    rec = store.create_identity("Cam")
    store.add_embedding_sample(rec.identity_id, vec(), MODEL)
    store.reset_database()
    assert store.list_identities() == []
    fresh = store.create_identity("Cam")  # name free again, schema intact
    assert store.get_identity(fresh.identity_id).sample_count == 0


def test_reset_survives_reopen(tmp_path):
    path = tmp_path / "identities.db"
    with IdentityStore(path) as store:
        rec = store.create_identity("Cam")
        store.add_embedding_sample(rec.identity_id, vec(), MODEL)
        store.reset_database()
    with IdentityStore(path) as store:
        assert store.list_identities() == []


# ---------------------------------------------------------------- concurrency

def test_concurrent_adds_from_threads(store):
    rec = store.create_identity("Cam")
    errors = []

    def add_many(base):
        try:
            for i in range(20):
                store.add_embedding_sample(rec.identity_id, vec(base * 100 + i), MODEL)
        except Exception as exc:  # noqa: BLE001 — test records any failure
            errors.append(exc)

    threads = [threading.Thread(target=add_many, args=(t,)) for t in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert store.get_identity(rec.identity_id).sample_count == 100
    assert len(store.get_embedding_samples(rec.identity_id)) == 100
