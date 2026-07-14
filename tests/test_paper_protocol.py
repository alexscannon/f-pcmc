"""T17 Phase 3 tests — the F-PCMC paper-protocol scorer + replay iterator.

Three layers (baselines/pcmc_sleep/PLAN.md Phase 3; owner Q&A 2026-07-14):

* [U] scorer math on the synthetic fixture world: the vectorized functions
  against a LITERAL per-image transliteration of the vendored loops with one
  patch per image — the executable documentation of the whole-image
  degenerate case (upstream lines cited in fpcmc_scorer.py) — plus rho
  gating / default-class behavior, their purity nan quirk, the
  contiguous-label guard, and the seeded eval-draw recipe.
* [U] the additive replay checkpoint-state iterator (owner Q8) on a fixture
  run: yields the task-0 state then every checkpoint state, each
  cross-checked against the log's own checkpoint record; exhausting it
  reproduces replay()'s final state exactly.
* [I][slow] the archived fpcmc_default/p2_seed42 cell: CIFAR row mapping is
  byte-identical to the pixel mirror's, the seeded draws reproduce
  p2_stream's recipe, all 44 reconstructed checkpoint states match their
  records, and scored records are sane. Skips cleanly without
  roots.env/live data (CLAUDE.md data contract).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from baselines.pcmc_sleep.fpcmc_scorer import (
    _unit_rows,
    aux_cluster_arrays,
    centroid_matrix,
    cifar_class_rows,
    cifar_eval_rows,
    classify,
    cluster_analytic,
    ltm_concepts,
    match_concepts,
    purity_score,
    score_checkpoint,
    supervise_cent_g,
    tier1_concepts,
    verify_checkpoint_record,
)
from baselines.pcmc_sleep.run_config import CLUST_SIZE, RHO, SUP_SIZE, TEST_SIZE
from fpcmc.config import FPCMCConfig, UmapConfig
from fpcmc.data import embeddings_available
from fpcmc.init import initialize_ltm
from fpcmc.replay import iter_checkpoint_states, read_log, replay
from fpcmc.rng import make_rng
from fpcmc.stream import StreamRunner
from fpcmc.thresholds import compute_global_prior
from tests.fixtures.vmf_world import Segment, VMFWorld

AVAILABLE, REASON = embeddings_available()

SEED = 1703


# ------------------------------------- literal transliterations (their loops)
#
# One patch per image: z has shape (1, D), so every `range(z.shape[0])`
# below is range(1). Line references: vendor/core/models/pcmc/.


def _their_supervise(Z, Y, ltm):
    """pcmc_layer.py:211-274, per-image loops kept literal."""
    Zn, Ln = _unit_rows(Z), _unit_rows(ltm)
    D_sum = 0.0
    for x in Zn:  # pass 1 (216-225)
        d = 1 - x[None, :] @ Ln.T
        close_ind = np.argmin(d, axis=1)
        D_sum += np.sum(d[range(1), close_ind]) / 1
    D = D_sum / len(Zn)

    sum_fz_pool = np.zeros((len(ltm), len(np.unique(Y))))
    for it, x in enumerate(Zn):  # pass 2 (233-253)
        d = 1 - x[None, :] @ Ln.T
        close_ind = np.argmin(d, axis=1)
        dist = d[range(1), close_ind]
        td = np.zeros(d.shape)
        td[range(1), close_ind] = np.exp(-1 * dist / D)
        fz = np.amax(td, axis=0)
        sum_fz_pool[:, int(Y[it])] += fz

    cent_g = np.copy(sum_fz_pool)  # 258-261
    for j in range(len(ltm)):
        cent_g[j, :] = cent_g[j, :] / (np.sum(cent_g[j, :]) + 1e-5)
    return cent_g


def _their_classify(Z, Y, ltm, cent_g, rho):
    """pcmc_layer.py:277-312 + pcmc.py:113-181, per-image loops literal."""
    Zn, Ln = _unit_rows(Z), _unit_rows(ltm)
    y_true = np.asarray(Y)
    num_classes = len(np.unique(y_true))
    v_l = np.zeros((len(Zn), num_classes))
    for it, x in enumerate(Zn):
        d = 1 - x[None, :] @ Ln.T
        close_ind = np.argmin(d, axis=1)
        votes = np.amax(cent_g[close_ind, :], axis=1).copy()
        votes[votes < rho] = 0  # 298
        vote_class = np.argmax(cent_g[close_ind, :], axis=1)
        for i in range(len(votes)):  # 302-303
            v_l[it, vote_class[i]] += votes[i]
    y_pred = np.argmax(v_l, axis=1)  # pcmc.py:148 (single layer)
    score = 100 * np.mean(y_true == y_pred)
    confusion = np.zeros((num_classes, num_classes))
    for cf in range(len(y_pred)):  # pcmc.py:151-154
        confusion[int(y_true[cf]), int(y_pred[cf])] += 1
    score_pc = [
        100 * confusion[i, i] / np.sum(confusion[i]) for i in range(num_classes)
    ]
    return score, score_pc, y_pred


# ----------------------------------------------------------- [U] scorer math


_WORLD = VMFWorld(seed=SEED, k_known=5, k_novel=1, kappa_known=400.0)


def _sup_test_split():
    names = _WORLD.known_names
    Z_sup = np.vstack([_WORLD.sample_class(n, 30, stream="pp_sup") for n in names])
    y_sup = np.repeat(np.arange(len(names)), 30)
    Z_test = np.vstack([_WORLD.sample_class(n, 20, stream="pp_test") for n in names])
    y_test = np.repeat(np.arange(len(names)), 20)
    # 6 centroids: the 5 known means + one never-supervised novel mean.
    centroids = np.vstack([_WORLD.true_means()[:5], _WORLD.true_mean("novel_00")])
    return Z_sup, y_sup, Z_test, y_test, centroids


def test_supervise_matches_their_loop():
    """The vectorized degenerate supervise equals a literal transliteration
    of their per-image loop (identical matched indices; cent_g to float
    accumulation-order tolerance)."""
    Z_sup, y_sup, _, _, centroids = _sup_test_split()
    ours = supervise_cent_g(Z_sup, y_sup, centroids)
    theirs = _their_supervise(Z_sup, y_sup, centroids)
    np.testing.assert_allclose(ours, theirs, rtol=1e-12, atol=1e-15)
    # Identical sparsity: an entry is nonzero iff some sup image of that
    # class matched that centroid — structure must agree exactly.
    assert np.array_equal(ours != 0, theirs != 0)


def test_classify_matches_their_loop():
    """The one-vote-per-image classify equals their loop bit-for-bit when
    fed the same cent_g (predictions, accuracy, per-class accuracies)."""
    Z_sup, y_sup, Z_test, y_test, centroids = _sup_test_split()
    cent_g = supervise_cent_g(Z_sup, y_sup, centroids)
    score, score_pc = classify(Z_test, y_test, centroids, cent_g, RHO)
    t_score, t_pc, t_pred = _their_classify(Z_test, y_test, centroids, cent_g, RHO)
    assert score == t_score
    assert score_pc == t_pc
    # And the fixture world is easy at kappa 400: sanity, not a pin.
    assert score > 80.0


def test_classify_rho_gate_and_default_class():
    """A matched concept whose best g-value is below rho contributes a zero
    vote; an all-zero vote row predicts class 0 (their argmax-of-zeros —
    released behavior, conform)."""
    centroids = np.eye(8)[:3]
    cent_g = np.array([[0.9, 0.1], [0.2, 0.8], [0.0, 0.0]])
    Z = np.eye(8)[[0, 1, 2]]
    y = np.array([0, 1, 1])  # image 2 matches the unsupervised concept
    score, score_pc = classify(Z, y, centroids, cent_g, rho=0.25)
    # predictions: [0, 1, 0] — the third falls back to class 0 and is wrong.
    assert score == pytest.approx(100 * 2 / 3)
    assert score_pc[0] == 100.0 and score_pc[1] == 50.0
    # Raising rho above 0.8 kills the genuine class-1 vote too.
    score_hi, _ = classify(Z, y, centroids, cent_g, rho=0.85)
    assert score_hi == pytest.approx(100 * 1 / 3)


def test_purity_score_verbatim_nan_quirk():
    """Their per-class purity averaging: a class that never wins a cluster
    divides 0/0 -> nan (released behavior, kept verbatim)."""
    acc, pc = purity_score([0, 0, 0, 1, 1], [0, 0, 0, 0, 0])
    assert acc == pytest.approx(3 / 5)
    assert pc[0] == pytest.approx(3 / 5)
    assert math.isnan(pc[1])
    acc2, pc2 = purity_score([0, 0, 0, 1, 1], [7, 7, 7, 2, 2])
    assert acc2 == 1.0 and pc2.tolist() == [1.0, 1.0]


def test_cluster_analytic_scores_the_partition():
    """Q9 analytic collapse: cluster = matched concept. On well-separated
    fixture classes with per-class centroids the partition is pure; note
    their scaling quirk (total x100, per-class raw fractions) is preserved."""
    _, _, Z_test, y_test, centroids = _sup_test_split()
    acc, pc = cluster_analytic(Z_test, y_test, centroids)
    assert acc == 100.0
    assert np.allclose(pc, 1.0)  # fractions, NOT x100 — their return shape


def test_label_contiguity_guard():
    Z = np.eye(8)[:4]
    with pytest.raises(ValueError, match="contiguous"):
        supervise_cent_g(Z, np.array([0, 0, 2, 2]), np.eye(8)[:2])
    with pytest.raises(ValueError, match="contiguous"):
        classify(Z, np.array([1, 1, 2, 2]), np.eye(8)[:2], np.zeros((2, 2)))


# --------------------------------------------------- [U] eval-set assembly


@dataclass(frozen=True)
class FakePhase:
    name: str
    group: str
    start: int
    end: int
    introduced_classes: tuple = field(default_factory=tuple)


@dataclass
class FakePool:
    subclass_names: list
    x: np.ndarray


T0 = ("a0", "a1")
PHASES = (
    FakePhase("heldout_00", "heldout", 0, 100, ("h0",)),
    FakePhase("near_00", "near", 100, 160, ("n0",)),
)


def _fake_pools(rows_per_class=6, d=4):
    classes = ["a0", "a1", "h0"]
    names = [c for c in classes for _ in range(rows_per_class)]
    rng = np.random.default_rng(0)
    make = lambda: FakePool(list(names), rng.standard_normal((len(names), d)))
    return {"ind_reference": make(), "ind_test": make()}


def test_cifar_eval_rows_recipe():
    """The CPU eval-set draw: deterministic seeded sup pick (sorted,
    without replacement), test = all rows unless oversized (then seeded),
    clustering set = per-class PREFIX of the selected test rows (Q6)."""
    pools = _fake_pools()
    kw = dict(sup_size=3, test_size=2, clust_size=1, seed=42)
    a = cifar_eval_rows(pools, T0, PHASES, 1, kw["seed"], sup_size=3,
                        test_size=2, clust_size=1)
    b = cifar_eval_rows(pools, T0, PHASES, 1, kw["seed"], sup_size=3,
                        test_size=2, clust_size=1)
    for k in a:
        assert np.array_equal(a[k], b[k]), k

    # Per class: sup rows drawn from that class's rows, ascending.
    for cls, label in (("a0", 0), ("a1", 1), ("h0", 2)):
        cls_rows = cifar_class_rows(pools["ind_reference"], cls)
        sup = a["sup_rows"][a["sup_y"] == label]
        assert np.isin(sup, cls_rows).all() and np.all(np.diff(sup) > 0)
        # The draw is the recorded recipe verbatim (parity contract §5).
        rng = make_rng(42, f"pcmc_sleep/sup/{cls}")
        pick = np.sort(rng.choice(cls_rows.size, size=3, replace=False))
        assert np.array_equal(sup, cls_rows[pick])
        # test: seeded 2-of-6 draw; clust: its 1-prefix.
        tst = a["test_rows"][a["test_y"] == label]
        clu = a["clust_rows"][a["clust_y"] == label]
        assert tst.size == 2 and np.array_equal(clu, tst[:1])

    # Labels contiguous over T0-then-introduction order; task 0 excludes h0.
    assert sorted(np.unique(a["sup_y"])) == [0, 1, 2]
    t0_only = cifar_eval_rows(pools, T0, PHASES, 0, 42, sup_size=3,
                              test_size=2, clust_size=1)
    assert sorted(np.unique(t0_only["sup_y"])) == [0, 1]


def test_aux_cluster_arrays_stream_prefix():
    """Aux clustering: CIFAR clustering subset + each visible synthetic
    class's first clust_size stream arrivals; None before any synthetic."""

    @dataclass
    class FakeManifest:
        true_class: np.ndarray

    @dataclass
    class FakeStream:
        t0_classes: tuple
        phases: tuple
        manifest: FakeManifest
        x: np.ndarray

    pools = _fake_pools()
    seq = ["a0"] * 100 + ["n0", "a1", "n0", "n0"] * 15
    stream = FakeStream(
        T0, PHASES, FakeManifest(np.array(seq, dtype=str)),
        np.arange(len(seq) * 4, dtype=float).reshape(len(seq), 4),
    )
    rows = cifar_eval_rows(pools, T0, PHASES, 2, 42, sup_size=3,
                           test_size=2, clust_size=2)
    assert aux_cluster_arrays(stream, pools, 1, rows, clust_size=2) is None
    aux = aux_cluster_arrays(stream, pools, 2, rows, clust_size=2)
    assert aux is not None
    X, y = aux
    # 3 CIFAR classes x 2 + 2 synthetic arrivals; synthetic label = 3 (next
    # contiguous cluster label); embeddings are the arrival rows themselves.
    assert X.shape[0] == y.size == 8
    first_n0 = np.flatnonzero(stream.manifest.true_class == "n0")[:2]
    np.testing.assert_array_equal(X[-2:], stream.x[first_n0])
    assert y[-2:].tolist() == [3, 3]


# ------------------------------------------- [U] replay checkpoint iterator


def _small_config(**kw) -> FPCMCConfig:
    base = dict(
        stm_capacity=8, n_mature=3, theta_promote=10, m_windows=2,
        window_W=100, T_merge=200, T_cluster=200, w_residual=100,
        umap=UmapConfig(dim=200), seed=42,
    )
    base.update(kw)
    return FPCMCConfig(**base)


_RUN_WORLD = VMFWorld(seed=SEED, k_known=4, k_novel=2,
                      kappa_novel=(600.0, 150.0))


def _fixture_run(config, log_path, checkpoint_steps=(99, 249, 399)):
    known = _RUN_WORLD.known_names

    def kn(n):
        base, rem = divmod(n, len(known))
        return {k: base + (1 if i < rem else 0) for i, k in enumerate(known)}

    schedule = [
        Segment(counts=kn(100)),
        Segment(counts={**kn(70), "novel_00": 30}),
        Segment(counts={**kn(50), "novel_00": 30, "novel_01": 20},
                distractors=tuple(range(10))),
        Segment(counts={**kn(60), "novel_00": 20, "novel_01": 20}),
    ]
    stream = _RUN_WORLD.make_stream(schedule)
    pool = _RUN_WORLD.t0_pool(n_per_class=80)
    store = initialize_ltm(pool.x, pool.labels, config)
    prior = compute_global_prior(store.ltm, config)
    runner = StreamRunner(config, store, prior, log_path=log_path,
                          checkpoint_steps=checkpoint_steps)
    runner.run(stream.x)
    return stream, pool, runner


def test_iter_checkpoint_states(tmp_path):
    """Owner Q8: the additive iterator yields the task-0 state then every
    checkpoint state, each matching the log's own checkpoint record (counts
    + full tau snapshot, exact); exhaustion equals replay()'s final state."""
    config = _small_config()
    log_path = tmp_path / "run.jsonl"
    stream, pool, runner = _fixture_run(config, log_path)
    records = read_log(log_path)
    cp = {r["step"]: r for r in records if r["type"] == "checkpoint"}
    assert sorted(cp) == [99, 249, 399]

    fresh = initialize_ltm(pool.x, pool.labels, config)
    prior = compute_global_prior(fresh.ltm, config)
    seen = []
    for step, store in iter_checkpoint_states(log_path, stream.x, fresh, prior):
        seen.append(step)
        if step == -1:
            assert len(store.ltm) == len(_RUN_WORLD.known_names)
            assert len(store.stm) == 0
        else:
            verify_checkpoint_record(store, cp[step])
    assert seen == [-1, 99, 249, 399]

    # Exhausting the iterator leaves the store at replay()'s final state.
    fresh2 = initialize_ltm(pool.x, pool.labels, config)
    prior2 = compute_global_prior(fresh2.ltm, config)
    replayed = replay(log_path, stream.x, fresh2, prior2)
    by_id = {c.concept_id: c for c in fresh.concepts}
    r_by_id = {c.concept_id: c for c in replayed.concepts}
    assert set(by_id) == set(r_by_id)
    for cid, c in by_id.items():
        r = r_by_id[cid]
        assert (c.status, c.provenance, c.match_count) == (
            r.status, r.provenance, r.match_count), cid
        assert c.ref_set.tobytes() == r.ref_set.tobytes(), cid
        assert c.centroid.tobytes() == r.centroid.tobytes(), cid
        assert (c.tau, c.kappa) == (r.tau, r.kappa), cid


def test_score_checkpoint_populations(tmp_path):
    """Q7: the record carries tier-1 headline fields + an ltm_only block;
    tier-1 is LTM ∪ mature STM exactly as the store's own predicate says."""
    config = _small_config()
    log_path = tmp_path / "run.jsonl"
    stream, pool, runner = _fixture_run(config, log_path)
    store = runner.store
    t1 = tier1_concepts(store)
    ltm = ltm_concepts(store)
    assert {c.concept_id for c in ltm} <= {c.concept_id for c in t1}
    for c in t1:
        assert c.status == "LTM" or c.match_count >= config.n_mature
    assert centroid_matrix(t1).shape == (len(t1), _RUN_WORLD.d)

    # Score against the fixture world as a fake "P2": build fake stream/pool
    # views over the fixture pools (labels contiguous by construction).
    # novel_00 plays a held-out CIFAR class (needs pool rows); novel_01
    # plays a synthetic class (enters via its stream arrivals only).
    known = _RUN_WORLD.known_names
    phases = (
        FakePhase("heldout_00", "heldout", 0, 200, ("novel_00",)),
        FakePhase("near_00", "near", 200, 400, ("novel_01",)),
    )

    @dataclass
    class FakeManifest:
        true_class: np.ndarray

    @dataclass
    class FakeStream:
        t0_classes: tuple
        phases: tuple
        manifest: FakeManifest
        x: np.ndarray

    # The eval universe at task 1 is known + novel_00, so both pools must
    # carry rows for all of them (the real ind_reference/ind_test hold all
    # 100 CIFAR classes — held-out classes included).
    eval_classes = known + ["novel_00"]
    ref = _RUN_WORLD.make_pool(eval_classes, 80, "pp_ref")
    tst = _RUN_WORLD.make_pool(eval_classes, 30, "pp_tst")
    pools = {
        "ind_reference": FakePool(list(ref.labels), ref.x),
        "ind_test": FakePool(list(tst.labels), tst.x),
    }
    fake = FakeStream(tuple(known), phases,
                      FakeManifest(np.asarray(stream.labels, dtype=str)),
                      stream.x)
    rec = score_checkpoint(store, fake, pools, task=2, seed=config.seed,
                           sup_size=40, test_size=20, clust_size=10)
    assert set(rec) >= {"class_acc", "class_pc_acc", "clust_acc",
                        "clust_pc_acc", "aux_clust_acc", "aux_clust_pc_acc",
                        "ltm_only", "n_tier1", "n_ltm", "n_stm"}
    assert rec["n_tier1"] == len(t1) and rec["n_ltm"] == len(ltm)
    assert 0.0 <= rec["class_acc"] <= 100.0
    assert 0.0 <= rec["ltm_only"]["class_acc"] <= 100.0
    assert rec["aux_clust_acc"] is not None  # novel_01 (synthetic) at task 2


# --------------------------------------------------- [I][slow] archived cell


@pytest.mark.slow
def test_archived_seed42_cell_parity_and_reconstruction():
    """The parity contract on the real archive (HANDOFF_PHASE3 §9): CIFAR
    row mapping byte-identical to the pixel mirror's, the seeded draws equal
    p2_stream's recipe outputs, all 44 reconstructed checkpoint states match
    their own checkpoint records exactly, and scored records are sane."""
    if not AVAILABLE:
        pytest.skip(REASON)
    from run_matrix import _init_store, default_out_root, embeddings_dir_for_encoder
    from fpcmc.data import load_all_pools
    from fpcmc.protocols import build_p2

    cell = default_out_root() / "fpcmc_default" / "p2_seed42"
    if not (cell / "events.jsonl").is_file():
        pytest.skip(f"archived cell not found: {cell}")

    records = read_log(cell / "events.jsonl")
    header = records[0]
    config = FPCMCConfig.from_yaml_text(__import__("yaml").safe_dump(header["config"]))
    pools = load_all_pools(embeddings_dir_for_encoder(config.encoder))
    stream = build_p2(config, config.seed, pools)
    assert list(stream.checkpoint_steps) == header["checkpoint_steps"]

    # --- eval-set byte-identity with the pixel mirror, where pixels exist.
    from baselines.pcmc_sleep.stream_mirror import P2PixelMirror, _data_root

    if (_data_root() / "cifar100" / "cifar-100-python").is_dir():
        mirror = P2PixelMirror(stream, pools)
        eval_classes = [str(c) for c in stream.t0_classes]
        for p in stream.phases:
            if p.group == "heldout":
                eval_classes += [str(c) for c in p.introduced_classes]
        assert len(eval_classes) == 100
        for cls in eval_classes:
            np.testing.assert_array_equal(
                cifar_class_rows(pools["ind_reference"], cls),
                mirror.cifar_rows("train", cls), err_msg=cls)
            np.testing.assert_array_equal(
                cifar_class_rows(pools["ind_test"], cls),
                mirror.cifar_rows("test", cls), err_msg=cls)

    # --- the real draws at the final task: sizes and recipe (p2_stream's
    # _cifar_eval_items reproduced through the same substreams).
    rows = cifar_eval_rows(pools, stream.t0_classes, stream.phases,
                           task=len(stream.phases), seed=config.seed)
    assert rows["sup_rows"].size == 100 * SUP_SIZE
    assert rows["test_rows"].size == 100 * TEST_SIZE  # all 100 test rows/class
    assert rows["clust_rows"].size == 100 * CLUST_SIZE
    aux = aux_cluster_arrays(stream, pools, len(stream.phases), rows)
    assert aux is not None and aux[0].shape[0] > rows["clust_rows"].size

    # --- full reconstruction: every checkpoint state matches its record.
    store, prior = _init_store(stream, pools, config)
    cp = {r["step"]: r for r in records if r["type"] == "checkpoint"}
    scored = {}
    for step, st in iter_checkpoint_states(
        cell / "events.jsonl", stream.x, store, prior
    ):
        if step == -1:
            assert len(st.ltm) == 80 and len(st.stm) == 0
            scored["t0"] = score_checkpoint(st, stream, pools, 0, config.seed)
        else:
            verify_checkpoint_record(st, cp[step])
    assert len(cp) == 44
    scored["final"] = score_checkpoint(
        st, stream, pools, len(stream.phases), config.seed
    )
    for rec in scored.values():
        assert 0.0 <= rec["class_acc"] <= 100.0
        assert 0.0 <= rec["clust_acc"] <= 100.0
        assert rec["n_tier1"] >= rec["n_ltm"] >= 80
    assert scored["final"]["n_ltm"] == cp[21537]["n_ltm"]
    assert scored["t0"]["aux_clust_acc"] is None
