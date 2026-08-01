# Identity storage format

The identity layer (`identity_lab/identity/`) is the canonical persistence
layer: a small repository-style library over one local SQLite file. No SQL
or SQLite types leak past `store.py`; callers see dataclasses and typed
exceptions only.

## Location

`~/Library/Application Support/IdentityLab/identities.db`
(tests and tools may point `IdentityStore` at any path).

The file contains biometric-derived data (face embeddings). It never leaves
the machine, is gitignored, and can be wiped either via
`IdentityStore.reset_database()` or by deleting the file.

## Schema (version 1)

```sql
schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT)

identities(
    id TEXT PRIMARY KEY,              -- uuid4 hex
    display_name TEXT NOT NULL,       -- unique, case-insensitive
    created_at TEXT NOT NULL,         -- ISO-8601 UTC
    enrollment_version INTEGER NOT NULL
)

embedding_samples(
    id TEXT PRIMARY KEY,              -- uuid4 hex
    identity_id TEXT NOT NULL REFERENCES identities(id) ON DELETE CASCADE,
    embedding BLOB NOT NULL,          -- see binary format below
    dim INTEGER NOT NULL,             -- element count (512 for buffalo_l)
    dtype TEXT NOT NULL,              -- always 'float32' in schema v1
    model_id TEXT NOT NULL,           -- e.g. 'buffalo_l'
    created_at TEXT NOT NULL          -- ISO-8601 UTC
)
```

Migrations are explicit and versioned: `store.py` applies missing versions
in order at open; a database whose recorded version is *newer* than the
code raises `SchemaMismatchError` and is never touched. Existing migrations
are never edited — schema changes add a new version.

## Embedding binary format

Defined in `identity_lab/identity/codec.py` (the only place embeddings
become bytes):

| Property | Value |
| --- | --- |
| dtype | IEEE-754 float32, 4 bytes per element |
| byte order | little-endian, always (numpy `<f4`), regardless of platform |
| layout | one C-contiguous 1-D vector; no header, no padding |
| blob length | exactly `dim * 4` bytes; `dim` lives in its own column |
| dimensionality | model-defined; 512 for buffalo_l (`w600k_r50`) |
| normalization | expected unit-L2-normalized (insightface `normed_embedding`); not enforced — consumers must normalize before cosine similarity if provenance is uncertain |
| model identifier | `model_id` column; embeddings from different model ids live in different vector spaces and are never comparable |

To read a blob by hand:
`numpy.frombuffer(blob, dtype='<f4')` and check `len == dim`.

## Validation & typed errors

- On write: arrays must be numpy, 1-D, float32, non-empty, all-finite
  (`MalformedEmbeddingError` otherwise). All samples of one identity must
  share `model_id` and `dim` (`ModelMismatchError`).
- On read: declared dtype must be supported, blob length must match
  `dim * 4`, values must be finite — corrupted rows raise
  `MalformedEmbeddingError` rather than being silently skipped.
- Not-a-database files raise `CorruptedDatabaseError`; unopenable paths and
  closed stores raise `DatabaseUnavailableError`; missing ids raise
  `IdentityNotFoundError`; duplicate names raise `DuplicateIdentityError`.

## Other behaviors

- Thread-safe via one internal lock (correctness over concurrency).
- `secure_delete` is ON: deleted rows are overwritten with zeros.
  `reset_database()` also VACUUMs so no embedding bytes linger in free pages.
- Samples return in insertion order.
- `enrollment_version` (currently 1) records which enrollment procedure
  produced an identity's samples, so future procedure changes can migrate
  or invalidate old enrollments knowingly.
