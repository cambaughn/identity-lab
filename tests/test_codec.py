"""Embedding codec: the documented binary format, enforced by tests."""

import numpy as np
import pytest

from identity_lab.identity.codec import decode_embedding, encode_embedding
from identity_lab.identity.errors import MalformedEmbeddingError


def test_encode_produces_little_endian_float32_bytes():
    v = np.array([1.0, -2.5, 3.25], dtype=np.float32)
    blob, dim = encode_embedding(v)
    assert dim == 3
    assert len(blob) == 3 * 4
    # Little-endian regardless of platform: reinterpret explicitly.
    assert np.array_equal(np.frombuffer(blob, dtype="<f4"), v)


def test_round_trip_is_byte_exact():
    rng = np.random.default_rng(42)
    v = rng.standard_normal(512).astype(np.float32)
    blob, dim = encode_embedding(v)
    out = decode_embedding(blob, dim, "float32")
    assert out.dtype == np.float32
    assert np.array_equal(out, v)


def test_decode_returns_writable_copy():
    blob, dim = encode_embedding(np.ones(4, dtype=np.float32))
    out = decode_embedding(blob, dim, "float32")
    out[0] = 99.0  # must not raise (frombuffer alone would be read-only)


def test_encode_rejects_non_contiguous_view_correctly():
    base = np.arange(20, dtype=np.float32)
    strided = base[::2]  # non-contiguous but valid values
    blob, dim = encode_embedding(strided)
    assert np.array_equal(decode_embedding(blob, dim, "float32"), strided)


@pytest.mark.parametrize("dtype", [np.float64, np.int32, np.float16])
def test_encode_rejects_wrong_dtypes(dtype):
    with pytest.raises(MalformedEmbeddingError):
        encode_embedding(np.zeros(8, dtype=dtype))


def test_encode_rejects_non_array_2d_empty_nonfinite():
    with pytest.raises(MalformedEmbeddingError):
        encode_embedding([1.0, 2.0])
    with pytest.raises(MalformedEmbeddingError):
        encode_embedding(np.zeros((2, 4), dtype=np.float32))
    with pytest.raises(MalformedEmbeddingError):
        encode_embedding(np.zeros(0, dtype=np.float32))
    bad = np.zeros(8, dtype=np.float32)
    bad[3] = np.inf
    with pytest.raises(MalformedEmbeddingError):
        encode_embedding(bad)


def test_decode_rejects_bad_inputs():
    blob, dim = encode_embedding(np.ones(8, dtype=np.float32))
    with pytest.raises(MalformedEmbeddingError):
        decode_embedding(blob, dim, "float64")        # unsupported dtype tag
    with pytest.raises(MalformedEmbeddingError):
        decode_embedding(blob, dim + 1, "float32")    # length mismatch
    with pytest.raises(MalformedEmbeddingError):
        decode_embedding(blob[:-1], dim, "float32")   # truncated
    with pytest.raises(MalformedEmbeddingError):
        decode_embedding(blob, 0, "float32")          # non-positive dim
    with pytest.raises(MalformedEmbeddingError):
        decode_embedding(blob, "8", "float32")        # dim not an int
