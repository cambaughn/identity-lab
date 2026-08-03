"""Matcher: normalization, similarity, top-k, threshold, margin, gallery.

Synthetic embeddings throughout — deterministic and webcam-free.
"""

import numpy as np
import pytest

from identity_lab.identity.matcher import (
    REASON_AMBIGUOUS,
    REASON_BELOW_THRESHOLD,
    REASON_MATCH,
    REASON_NO_EMBEDDING,
    REASON_NO_IDENTITIES,
    GalleryIdentity,
    Matcher,
    build_gallery,
    cosine_similarity,
    normalize,
    top_k_mean,
)
from identity_lab.identity.store import IdentityStore

DIM = 512


def unit(seed, dim=DIM):
    v = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def near(base, seed, noise=0.05):
    v = base + np.random.default_rng(seed).normal(0, noise, base.size).astype(np.float32)
    return v / np.linalg.norm(v)


def gallery_identity(identity_id, name, base_seed, n=5, noise=0.05):
    base = unit(base_seed)
    rows = np.vstack([near(base, base_seed * 1000 + i, noise) for i in range(n)])
    return GalleryIdentity(identity_id=identity_id, display_name=name, embeddings=rows)


CAM_BASE = unit(1)
ALE_BASE = unit(2)
CAM = gallery_identity("id-cam", "Cam", 1)
ALE = gallery_identity("id-ale", "Ale", 2)

DEFAULTS = dict(threshold=0.40, margin=0.08, top_k=3)


# ---------------------------------------------------------------- primitives

def test_normalize_produces_unit_vector():
    out = normalize(np.array([3.0, 4.0], dtype=np.float32))
    assert out is not None
    assert np.isclose(np.linalg.norm(out), 1.0)
    assert np.allclose(out, [0.6, 0.8])


def test_normalize_rejects_zero_and_nonfinite():
    assert normalize(np.zeros(4, dtype=np.float32)) is None
    bad = np.ones(4, dtype=np.float32)
    bad[1] = np.nan
    assert normalize(bad) is None
    assert normalize(np.zeros(0, dtype=np.float32)) is None


def test_normalize_accepts_other_dtypes():
    out = normalize(np.array([2, 0], dtype=np.int64))
    assert out is not None and out.dtype == np.float32
    assert np.allclose(out, [1.0, 0.0])


def test_cosine_similarity_basics():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert cosine_similarity(a, a) == pytest.approx(1.0)
    assert cosine_similarity(a, b) == pytest.approx(0.0)
    assert cosine_similarity(a, -a) == pytest.approx(-1.0)
    assert cosine_similarity(a * 7, a * 0.1) == pytest.approx(1.0)  # scale-free
    assert cosine_similarity(a, np.zeros(2, dtype=np.float32)) == 0.0


def test_top_k_mean_selection():
    sims = np.array([0.1, 0.9, 0.5, 0.7], dtype=np.float32)
    assert top_k_mean(sims, 1) == pytest.approx(0.9)
    assert top_k_mean(sims, 2) == pytest.approx(0.8)
    assert top_k_mean(sims, 10) == pytest.approx(sims.mean())  # k > n: all
    assert top_k_mean(np.array([]), 3) == 0.0
    assert top_k_mean(sims, 0) == pytest.approx(0.9)  # clamped to 1


# ---------------------------------------------------------------- matching

def test_empty_gallery_never_guesses():
    result = Matcher([]).match(unit(1), **DEFAULTS)
    assert result.is_known is False
    assert result.reason == REASON_NO_IDENTITIES
    assert result.display_name is None and result.similarity is None


def test_zero_embedding_never_guesses():
    result = Matcher([CAM]).match(np.zeros(DIM, dtype=np.float32), **DEFAULTS)
    assert result.is_known is False
    assert result.reason == REASON_NO_EMBEDDING


def test_single_identity_match():
    probe = near(CAM_BASE, 42)
    result = Matcher([CAM]).match(probe, **DEFAULTS)
    assert result.is_known is True
    assert result.display_name == "Cam"
    assert result.reason == REASON_MATCH
    # noise=0.05/element in 512-D adds a ~1.1-norm noise vector, so ~0.5
    # cosine between same-identity samples — realistic, and above threshold.
    assert result.similarity > 0.45
    assert result.second_best is None  # margin not applicable


def test_single_identity_below_threshold_reports_candidate():
    stranger = unit(99)
    result = Matcher([CAM]).match(stranger, **DEFAULTS)
    assert result.is_known is False
    assert result.reason == REASON_BELOW_THRESHOLD
    assert result.display_name is None          # no leak into the label
    assert result.best is not None              # ...but debug shows it
    assert result.similarity == pytest.approx(result.best.score)
    assert result.best.score < 0.4


def test_two_identities_picks_correct_one():
    probe = near(ALE_BASE, 7)
    result = Matcher([CAM, ALE]).match(probe, **DEFAULTS)
    assert result.is_known and result.display_name == "Ale"
    assert result.second_best is not None
    assert result.second_best.display_name == "Cam"
    assert result.best.score > result.second_best.score


def test_margin_rejects_ambiguous_probe():
    # Two near-duplicate identities: any probe close to one is close to both.
    twin_a = gallery_identity("a", "TwinA", 5, noise=0.01)
    twin_b = GalleryIdentity("b", "TwinB", twin_a.embeddings + 0.001)
    probe = near(unit(5), 11, noise=0.01)
    result = Matcher([twin_a, twin_b]).match(
        probe, threshold=0.40, margin=0.08, top_k=3
    )
    assert result.is_known is False
    assert result.reason == REASON_AMBIGUOUS
    assert result.best is not None and result.second_best is not None
    assert abs(result.best.score - result.second_best.score) < 0.08


def test_margin_zero_disables_ambiguity_check():
    twin_a = gallery_identity("a", "TwinA", 5, noise=0.01)
    twin_b = GalleryIdentity("b", "TwinB", twin_a.embeddings + 0.001)
    probe = near(unit(5), 11, noise=0.01)
    result = Matcher([twin_a, twin_b]).match(
        probe, threshold=0.40, margin=0.0, top_k=3
    )
    assert result.is_known is True


def test_threshold_extremes():
    probe = near(CAM_BASE, 42)
    high = Matcher([CAM]).match(probe, threshold=0.99, margin=0.0, top_k=3)
    assert high.is_known is False and high.reason == REASON_BELOW_THRESHOLD
    low = Matcher([CAM]).match(unit(99), threshold=0.05, margin=0.0, top_k=3)
    # Even a stranger may clear an absurdly low threshold — that's the
    # slider doing exactly what it says. Never a *forced* guess though:
    assert low.reason in (REASON_MATCH, REASON_BELOW_THRESHOLD)


def test_top_k_uses_best_samples():
    # Identity with 4 off-pose samples and 1 dead-on sample: top_k=1 should
    # score higher than top_k=5 (which averages the weak ones in).
    base = unit(31)
    rows = np.vstack([near(base, i, noise=0.6) for i in range(4)] + [base])
    ident = GalleryIdentity("x", "X", rows)
    r1 = Matcher([ident]).match(base, threshold=0.0, margin=0.0, top_k=1)
    r5 = Matcher([ident]).match(base, threshold=0.0, margin=0.0, top_k=5)
    assert r1.similarity == pytest.approx(1.0, abs=1e-5)
    assert r1.similarity > r5.similarity


# ---------------------------------------------------------------- gallery

@pytest.fixture
def store(tmp_path):
    s = IdentityStore(tmp_path / "identities.db")
    yield s
    s.close()


def fill(store, name, base_seed, n=5, model="buffalo_l"):
    rec = store.create_identity(name)
    base = unit(base_seed)
    for i in range(n):
        store.add_embedding_sample(rec.identity_id, near(base, base_seed * 100 + i), model)
    return rec


def test_build_gallery_loads_and_normalizes(store):
    fill(store, "Cam", 1)
    rec = store.create_identity("Scaled")
    scaled = (unit(3) * 25.0).astype(np.float32)  # stored un-normalized
    store.add_embedding_sample(rec.identity_id, scaled, "buffalo_l")
    gallery, warnings = build_gallery(store, "buffalo_l")
    assert warnings == []
    assert {g.display_name for g in gallery} == {"Cam", "Scaled"}
    for g in gallery:
        norms = np.linalg.norm(g.embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)


def test_build_gallery_skips_empty_and_wrong_model(store):
    fill(store, "Cam", 1)
    store.create_identity("Empty")
    fill(store, "OldModel", 4, model="ancient_model")
    gallery, warnings = build_gallery(store, "buffalo_l")
    assert [g.display_name for g in gallery] == ["Cam"]
    assert any("Empty" in w for w in warnings)
    assert any("OldModel" in w and "ancient_model" in w for w in warnings)


def test_build_gallery_survives_corrupted_identity(store, tmp_path):
    import sqlite3

    fill(store, "Cam", 1)
    bad = fill(store, "Corrupt", 2)
    with sqlite3.connect(tmp_path / "identities.db") as conn:
        conn.execute(
            "UPDATE embedding_samples SET embedding = X'00' "
            "WHERE identity_id = ?",
            (bad.identity_id,),
        )
    gallery, warnings = build_gallery(store, "buffalo_l")
    assert [g.display_name for g in gallery] == ["Cam"]
    assert any("Corrupt" in w for w in warnings)


def test_live_reload_after_enrollment_change(store):
    fill(store, "Cam", 1)
    gallery1, _ = build_gallery(store, "buffalo_l")
    matcher1 = Matcher(gallery1)
    ale_probe = near(ALE_BASE, 8)
    assert matcher1.match(ale_probe, **DEFAULTS).is_known is False

    fill(store, "Ale", 2)  # enroll a second person
    gallery2, _ = build_gallery(store, "buffalo_l")
    matcher2 = Matcher(gallery2)
    result = matcher2.match(ale_probe, **DEFAULTS)
    assert result.is_known and result.display_name == "Ale"
    # The old snapshot is unchanged (snapshot semantics, reload is explicit)
    assert matcher1.match(ale_probe, **DEFAULTS).is_known is False
    assert matcher2.identity_count == 2 and matcher2.sample_count == 10
