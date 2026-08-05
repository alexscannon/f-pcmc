"""T17 Phase 3: the F-PCMC paper-protocol scorer (evaluation parity).

Applies the PCMC paper's Eq. 6-7 labeled patch-vote classification and
clustering purity to F-PCMC concept states as the WHOLE-IMAGE DEGENERATE
CASE of the vendored implementation: one "patch" per image — its CLS
embedding — with concept centroids in the LTM-centroid role. Eval-side
supervised BY DESIGN: labels live here and never flow into ``fpcmc/`` (the
no-leakage AST tests keep it that way). Runs in the pinned root CPU env, no
new dependencies.

Owner decisions implemented (Phase 3 Q&A 2026-07-14, verbatim in PLAN.md):
  Q6  clustering eval subsampled to ``run_config.CLUST_SIZE`` per class (the
      first CLUST_SIZE of each class's seeded test draw / first CLUST_SIZE
      stream arrivals per synthetic class); classification stays
      SUP_SIZE/TEST_SIZE. Same constants as the PCMC in-run loaders.
  Q7  every checkpoint scored on BOTH populations — tier-1 (LTM ∪ mature
      STM, the store's own frozen maturity-only predicate) as the headline,
      LTM-only alongside in the same record.
  Q9  clustering = the ANALYTIC COLLAPSE: cluster assignment IS the matched
      concept, purity via their purity_score math verbatim. Measured
      (PLAN.md Phase 3 Q&A): their SpectralClustering on the degenerate
      binary jaccard affinity is NOT equivalent — it arbitrarily merges
      concept groups when concepts > 2×classes and arbitrarily splits
      identical-affinity groups when concepts < 2×classes (F-PCMC's actual
      regime), scrambling per-class purity and destabilizing across seeds.
      The binary affinity already IS a partition; we score that partition.
  Q10 this module lives in baselines/pcmc_sleep/; artifacts go under
      ``${DATA_ROOT}/evaluation/f_pcmc_runs/pcmc_sleep/paper_protocol/``.

Degenerate-case collapse, function by function (line references into
``baselines/pcmc_sleep/vendor/core/models/pcmc/``, byte-identical @ e77f5f7;
with one patch per image, every per-image patch loop has length 1):

* supervise (pcmc_layer.py:211-274) -> ``supervise_cent_g``. Pass 1: D =
  mean over images of (mean over patches of nearest-centroid cosine
  distance) = the plain mean nearest-concept distance (z.shape[0] == 1 at
  pcmc_layer.py:221). Pass 2: the sparse ``td`` matrix (248-249) holds one
  entry exp(-dist/D) at (patch 0, matched centroid); its column max ``fz``
  (250) is that same one-hot vector, accumulated into the raw-integer label
  column (253) — hence the contiguous-label invariant, asserted here. Row
  normalization with +1e-5 (260-261) is kept verbatim.
* classify (pcmc_layer.py:277-312, vote assembly pcmc.py:106-181) ->
  ``classify``. One patch => one vote per image: weight = max over classes
  of cent_g[matched] zeroed below rho (297-298), class = argmax (300); the
  per-image vote vector v_l has a single nonzero, so prediction = the
  matched concept's argmax class when its weight clears rho, else class 0
  (their argmax over an all-zero row — released behavior, conform).
  Accuracy ×100 and per-class accuracy from the confusion matrix
  (pcmc.py:151-179).
* embed (pcmc_layer.py:314-323) -> ``match_concepts``: the per-image "set of
  matched centroids" is the singleton {argmin distance}.
* cluster (pcmc.py:221-268) -> ``cluster_analytic`` (owner Q9): with
  singleton sets, their jaccard (184-192) is binary — 1 iff same matched
  concept — so the affinity partitions the eval set by matched concept; the
  analytic collapse scores that partition with their purity_score
  (pcmc.py:194-217, copied verbatim below including the per-class averaging
  whose 0/0 yields nan entries — serialized as JSON null, like the Phase 2
  driver's records). Their ``accu_total*100, accu_perclass`` scaling quirk
  (total ×100, per-class raw fractions vs classify's ×100 both) is kept.

Eval sets are byte-identical to the PCMC side's (§5 parity contract): the
CIFAR row selection + seeded draws reproduce ``p2_stream.P2UPLStream.
_cifar_eval_items`` exactly (same ``fpcmc.rng.make_rng`` substreams), with
pickle-row -> pool-row identity proven in Phase 0.1 (0 mismatches over all
63,326 rows). Ties in argmin resolve to the first (lowest) index in both
numpy (their supervise) and the torch path (their classify/embed);
population matrices are built in store registration order so the resolution
is deterministic here too.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics.cluster import contingency_matrix

from baselines.pcmc_sleep.run_config import (
    CLUST_SIZE,
    RHO,
    SUP_SIZE,
    TEST_SIZE,
    cluster_label_map,
    eval_classes_at,
    eval_label_map,
)
from fpcmc.rng import make_rng

_EPS = 1e-12


# --------------------------------------------------------------- linear math


def _unit_rows(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), _EPS)


def match_concepts(
    Z: np.ndarray, centroids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest concept per image under 1 - cosine similarity.

    The degenerate ``embed``/matching step (pcmc_layer.py:218-221, 292-295,
    314-323: ``1 - pairwise_cosine_similarity`` then argmin). Returns
    (matched index, matched distance); argmin ties -> first index.
    """
    d = 1.0 - _unit_rows(Z) @ _unit_rows(centroids).T
    close = np.argmin(d, axis=1)
    return close, d[np.arange(d.shape[0]), close]


def _check_labels(y: np.ndarray) -> int:
    """Their supervise/classify index score columns by the RAW integer label
    (pcmc_layer.py:253, 282), so labels must be exactly 0..C-1 with every
    class present — a violated invariant fails silently with garbage
    columns upstream; here it fails loudly."""
    classes = np.unique(y)
    n = int(classes.size)
    if not np.array_equal(classes, np.arange(n)):
        raise ValueError(f"eval labels must be contiguous 0..C-1; got {classes}")
    return n


def supervise_cent_g(
    Z_sup: np.ndarray, y_sup: np.ndarray, centroids: np.ndarray
) -> np.ndarray:
    """Degenerate supervise (pcmc_layer.py:211-274): the per-concept
    class-association matrix ``cent_g`` (their g values, Eq. 6)."""
    y_sup = np.asarray(y_sup)
    n_classes = _check_labels(y_sup)
    close, dist = match_concepts(Z_sup, centroids)
    D = float(dist.mean())  # pcmc_layer.py:216-225 with one patch per image
    sum_fz = np.zeros((centroids.shape[0], n_classes))
    # pcmc_layer.py:248-253: one exp(-dist/D) entry per image at its matched
    # centroid, accumulated into the image's label column.
    np.add.at(sum_fz, (close, y_sup), np.exp(-dist / D))
    # pcmc_layer.py:258-261: rows normalized by (row sum + 1e-5).
    return sum_fz / (sum_fz.sum(axis=1, keepdims=True) + 1e-5)


def classify(
    Z_test: np.ndarray,
    y_test: np.ndarray,
    centroids: np.ndarray,
    cent_g: np.ndarray,
    rho: float = RHO,
) -> tuple[float, list[float]]:
    """Degenerate classify (pcmc_layer.py:277-312 + pcmc.py:106-181):
    (accuracy×100, per-class accuracy×100 list)."""
    y_test = np.asarray(y_test)
    n_classes = _check_labels(y_test)
    if cent_g.shape[1] != n_classes:
        raise ValueError("cent_g class axis does not match the test labels")
    close, _ = match_concepts(Z_test, centroids)
    g = cent_g[close]
    votes = g.max(axis=1)
    vote_class = g.argmax(axis=1)
    votes = np.where(votes < rho, 0.0, votes)  # pcmc_layer.py:298
    # One patch => v_l has a single (possibly zero) entry per image
    # (pcmc_layer.py:302-303); argmax of an all-zero row is class 0
    # (pcmc.py:123/148) — conform.
    v_l = np.zeros((Z_test.shape[0], n_classes))
    v_l[np.arange(v_l.shape[0]), vote_class] = votes
    y_pred = v_l.argmax(axis=1)

    score = 100.0 * float(np.mean(y_test == y_pred))
    # pcmc.py:151-155, 179: per-class accuracy from the confusion matrix.
    cm = np.zeros((n_classes, n_classes))
    for t, p in zip(y_test, y_pred):
        cm[int(t), int(p)] += 1
    score_pc = [100.0 * cm[i, i] / np.sum(cm[i]) for i in range(n_classes)]
    return score, score_pc


def purity_score(y_true, y_pred):
    """VERBATIM math of their purity_score (pcmc.py:194-217), including the
    per-class averaging whose 0/0 (a class that never wins a cluster)
    yields nan entries — released behavior, tolerated downstream."""
    cm = contingency_matrix(y_true, y_pred)
    num_classes = len(np.unique(y_true))
    cluster_labs = np.argmax(cm, axis=0)
    class_perf = np.zeros(num_classes)
    class_counts = np.zeros(num_classes)
    for i, lab in enumerate(cluster_labs):
        class_perf[lab] += cm[lab, i] / np.sum(cm[:, i])
        class_counts[lab] += 1
    with np.errstate(invalid="ignore"):
        for i in range(len(class_counts)):
            class_perf[i] /= class_counts[i]
    acc, pc_acc = np.sum(np.amax(cm, axis=0)) / np.sum(cm), class_perf
    return acc, pc_acc


def cluster_analytic(
    Z: np.ndarray, y: np.ndarray, centroids: np.ndarray
) -> tuple[float, np.ndarray]:
    """Owner Q9: the analytic collapse of their cluster() (pcmc.py:221-268).

    Cluster assignment = matched concept index (the partition the binary
    singleton-set jaccard affinity already encodes); purity via their
    purity_score verbatim. Scaling matches their return exactly:
    ``accu_total*100, accu_perclass`` (per-class left as raw fractions).
    """
    close, _ = match_concepts(Z, centroids)
    acc, pc = purity_score(np.asarray(y), close)
    return 100.0 * float(acc), pc


# -------------------------------------------------- eval-set assembly (CPU)


def cifar_class_rows(pool, cls: str) -> np.ndarray:
    """Ascending pool rows of a CIFAR class == canonical pickle rows ==
    ``P2PixelMirror.cifar_rows`` (row-by-row alignment proven in Phase 0.1;
    asserted against the mirror in the [I] tests)."""
    labels = np.asarray(pool.subclass_names, dtype=str)
    return np.flatnonzero(labels == str(cls))


def cifar_eval_rows(
    pools: dict,
    t0_classes,
    phases,
    task: int,
    seed: int,
    *,
    sup_size: int = SUP_SIZE,
    test_size: int = TEST_SIZE,
    clust_size: int = CLUST_SIZE,
) -> dict:
    """CPU reproduction of ``P2UPLStream._cifar_eval_items`` (byte-identical
    draws: same substreams through fpcmc.rng.make_rng) + the Q6 clustering
    subset (first ``clust_size`` of each class's selected test rows).

    Returns pool-row indices and contiguous labels:
    ``{"sup_rows", "sup_y", "test_rows", "test_y", "clust_rows", "clust_y"}``
    with sup rows into ``ind_reference`` and test/clust rows into
    ``ind_test``.
    """
    ref, tst = pools["ind_reference"], pools["ind_test"]
    cifar, _ = eval_classes_at(t0_classes, phases, task)
    eval_map = eval_label_map(t0_classes, phases)
    sup_rows, sup_y, test_rows, test_y = [], [], [], []
    clust_rows, clust_y = [], []
    for cls in cifar:
        label = eval_map[cls]
        train_rows = cifar_class_rows(ref, cls)
        rng = make_rng(seed, f"pcmc_sleep/sup/{cls}")
        pick = np.sort(rng.choice(train_rows.size, size=sup_size, replace=False))
        sup_rows.append(train_rows[pick])
        sup_y.append(np.full(sup_size, label))

        rows = cifar_class_rows(tst, cls)
        if test_size < rows.size:
            rng = make_rng(seed, f"pcmc_sleep/test/{cls}")
            pick = np.sort(rng.choice(rows.size, size=test_size, replace=False))
            rows = rows[pick]
        test_rows.append(rows)
        test_y.append(np.full(rows.size, label))
        clust_rows.append(rows[:clust_size])  # Q6: prefix of the test draw
        clust_y.append(np.full(min(clust_size, rows.size), label))
    return {
        "sup_rows": np.concatenate(sup_rows),
        "sup_y": np.concatenate(sup_y),
        "test_rows": np.concatenate(test_rows),
        "test_y": np.concatenate(test_y),
        "clust_rows": np.concatenate(clust_rows),
        "clust_y": np.concatenate(clust_y),
    }


def aux_cluster_arrays(
    stream, pools, task: int, rows: dict, *, clust_size: int = CLUST_SIZE
) -> tuple[np.ndarray, np.ndarray] | None:
    """The Q3 auxiliary synthetic-inclusive clustering set, Q6-sized: the
    CIFAR clustering subset plus, per synthetic class visible at ``task``,
    its first ``clust_size`` stream arrivals (stream order; the arrival
    embeddings themselves — the PCMC side feeds the same stream-seen
    images). None while no synthetic class is visible, mirroring
    ``P2UPLStream.cluster_loader``."""
    _, synth = eval_classes_at(stream.t0_classes, stream.phases, task)
    if not synth:
        return None
    cmap = cluster_label_map(stream.t0_classes, stream.phases)
    true_class = np.asarray(stream.manifest.true_class, dtype=str)
    X = [np.asarray(pools["ind_test"].x)[rows["clust_rows"]]]
    y = [rows["clust_y"]]
    for cls in synth:
        steps = np.flatnonzero(true_class == str(cls))[:clust_size]
        X.append(np.asarray(stream.x)[steps])
        y.append(np.full(steps.size, cmap[str(cls)]))
    return np.concatenate(X), np.concatenate(y)


# ------------------------------------------------- concept populations (Q7)


def tier1_concepts(store) -> list:
    """LTM ∪ mature STM in registration order — exactly the tier-1 view
    ``ConceptStore.route`` answers with (fpcmc/concepts.py:556), through the
    store's own frozen maturity-only predicate ``_tier1_stm`` (maturity and
    ONLY maturity; read its docstring before "improving" this)."""
    return [c for c in store.concepts if c.status == "LTM" or store._tier1_stm(c)]


def ltm_concepts(store) -> list:
    """The literal analog of their ``self.ltm`` buffer: status == LTM only."""
    return list(store.ltm)


def centroid_matrix(concepts) -> np.ndarray:
    """(C, D) concept centroids (Concept.centroid: L2-normalized; EMA for
    STM, frozen for LTM), in the given (registration) order."""
    return np.stack([c.centroid for c in concepts])


# ------------------------------------------------------ checkpoint scoring


def score_population(
    centroids: np.ndarray,
    pools: dict,
    rows: dict,
    aux: tuple[np.ndarray, np.ndarray] | None,
    rho: float = RHO,
) -> dict:
    """One population's full paper-protocol record fields at one checkpoint."""
    ref_x = np.asarray(pools["ind_reference"].x)
    tst_x = np.asarray(pools["ind_test"].x)
    cent_g = supervise_cent_g(ref_x[rows["sup_rows"]], rows["sup_y"], centroids)
    class_acc, class_pc = classify(
        tst_x[rows["test_rows"]], rows["test_y"], centroids, cent_g, rho
    )
    clust_acc, clust_pc = cluster_analytic(
        tst_x[rows["clust_rows"]], rows["clust_y"], centroids
    )
    out = {
        "class_acc": class_acc,
        "class_pc_acc": class_pc,
        "clust_acc": clust_acc,
        "clust_pc_acc": clust_pc,
        "aux_clust_acc": None,
        "aux_clust_pc_acc": None,
    }
    if aux is not None:
        aux_acc, aux_pc = cluster_analytic(aux[0], aux[1], centroids)
        out["aux_clust_acc"] = aux_acc
        out["aux_clust_pc_acc"] = aux_pc
    return out


def score_checkpoint(
    store,
    stream,
    pools,
    task: int,
    seed: int,
    rho: float = RHO,
    *,
    sup_size: int = SUP_SIZE,
    test_size: int = TEST_SIZE,
    clust_size: int = CLUST_SIZE,
) -> dict:
    """Both Q7 populations at one reconstructed checkpoint state: tier-1
    fields at the top level (the headline), LTM-only nested under
    ``"ltm_only"``, plus population sizes for context. Size overrides exist
    for fixture-scale tests; production scoring uses the run_config
    defaults (the Q6 single constant set)."""
    rows = cifar_eval_rows(
        pools, stream.t0_classes, stream.phases, task, seed,
        sup_size=sup_size, test_size=test_size, clust_size=clust_size,
    )
    aux = aux_cluster_arrays(stream, pools, task, rows, clust_size=clust_size)
    t1 = tier1_concepts(store)
    record = score_population(centroid_matrix(t1), pools, rows, aux, rho)
    ltm = ltm_concepts(store)
    record["ltm_only"] = score_population(centroid_matrix(ltm), pools, rows, aux, rho)
    record["n_tier1"] = len(t1)
    record["n_ltm"] = len(ltm)
    record["n_stm"] = len(store.stm)
    return record


def verify_checkpoint_record(store, rec: dict) -> None:
    """Cross-check a reconstructed state against its live checkpoint record
    (counts + the full per-concept tau snapshot; exact — replay is
    bit-reproducible). Raises AssertionError on any drift."""
    assert len(store.ltm) == rec["n_ltm"], (len(store.ltm), rec["n_ltm"])
    assert len(store.stm) == rec["n_stm"], (len(store.stm), rec["n_stm"])
    taus = rec["taus"]
    ids = {c.concept_id for c in store.concepts}
    assert ids == set(taus), ids ^ set(taus)
    for c in store.concepts:
        entry = taus[c.concept_id]
        assert c.status == entry["status"], c.concept_id
        for attr, key in (("tau", "tau"), ("tau_vmf", "tau_vmf")):
            got = getattr(c, attr)
            want = entry[key]
            if want is None:
                assert not np.isfinite(got), (c.concept_id, attr, got)
            else:
                assert got == want, (c.concept_id, attr, got, want)
