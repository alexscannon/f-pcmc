"""Synthetic vMF test world — the fixture behind every F-PCMC unit test (T1).

A `VMFWorld` is a small, fully analytic universe of classes on the unit sphere
in D=32 (default) dimensions. Each class is a von Mises-Fisher distribution
vMF(mu_c, kappa_c). Later tasks (T2-T15) consume this module UNMODIFIED, so its
contract is spelled out here:

Geometry
  - Class mean directions are constructed EQUIANGULARLY: every pair of means
    sits at exactly `separation_deg` degrees (supported range (0, 90]). The
    construction is m_i = sqrt(c)*u + sqrt(1-c)*e_i with c = cos(separation),
    u ⟂ e_i an orthonormal frame, which gives m_i . m_j = c exactly for i != j.
    Requires k_known + k_novel + n_burst <= d - 1.
  - Per-class concentration kappa is configurable per group (scalar or one
    value per class). Tighter class == larger kappa. "Burst" classes are just
    ordinary vMF classes conventionally given a high kappa; they exist so
    outlier-burst scenarios (golden stream, T8's recurrence-criterion case)
    can sample an arbitrary number of near-duplicate points.
  - Distractors are NOT vMF classes: `distractor_point(i)` is a single
    uniformly random unit vector, labeled "distractor_{i:02d}" in streams.
    They model one-off outliers (each appears once, never recurs).

Determinism
  - Every draw flows through fpcmc.rng.make_rng(seed, stream) with a named
    substream per (purpose, class): sampling is a PURE FUNCTION of
    (world seed, stream label, class name, n). Same seed => bit-identical
    arrays, across calls and across world instances. There is no hidden state;
    calling any method never perturbs any other draw.
  - Corollary: the same class sampled under two different stream labels gives
    two independent samples (that is how t0/test/novel pools and each stream
    segment stay disjoint).

Class naming
  - Known classes:  "known_00" ... "known_{k_known-1:02d}"
  - Novel classes:  "novel_00" ...
  - Burst classes:  "burst_00" ...
  - Distractors:    "distractor_00" ... (stream labels only, not vMF classes)

Streams
  - `make_stream(schedule)` takes a sequence of `Segment`s and concatenates
    them. Each segment draws `counts[name]` fresh samples per class (plus any
    one-off `distractors` by index) and, if `shuffle=True`, interleaves them
    deterministically. `shuffle=False` keeps the block order (sorted class
    name, then distractors) — that is how a contiguous outlier burst is laid
    down. The returned `Stream` carries ground-truth labels and per-step
    segment ids.

Analytic helpers
  - `true_mean(name)`, `true_means()` / `true_mean_class_names()`,
    `true_pairwise_angles_deg()`, `kappa(name)` — for tests that compare
    empirical estimates against ground truth.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from fpcmc.rng import make_rng

_EPS = 1e-12


@dataclass(frozen=True)
class Segment:
    """One stream segment: per-class sample counts + optional one-off distractors.

    counts:      class name -> number of fresh samples of that class.
    distractors: indices of one-off distractor points to include (each
                 contributes exactly one embedding labeled "distractor_{i:02d}").
    shuffle:     True  -> deterministic interleave of the segment's points;
                 False -> contiguous blocks (classes in sorted-name order,
                          then distractors) — use for outlier bursts.
    """

    counts: Mapping[str, int]
    distractors: tuple[int, ...] = ()
    shuffle: bool = True


@dataclass(frozen=True)
class FixturePool:
    """A labeled sample pool. x: (N, D) float64 unit rows; labels: (N,) <U str."""

    x: np.ndarray
    labels: np.ndarray


@dataclass(frozen=True)
class Stream:
    """A generated stream. x: (N, D); labels: (N,) ground-truth class names;
    segment_ids: (N,) index into the schedule that produced each step."""

    x: np.ndarray
    labels: np.ndarray
    segment_ids: np.ndarray


def _kappa_list(kappa: float | Sequence[float], k: int, what: str) -> list[float]:
    if isinstance(kappa, (int, float)):
        return [float(kappa)] * k
    kappas = [float(v) for v in kappa]
    if len(kappas) != k:
        raise ValueError(f"{what}: expected {k} kappa values, got {len(kappas)}")
    return kappas


def sample_vmf(mu: np.ndarray, kappa: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw n samples from vMF(mu, kappa) on S^{d-1} (Wood 1994 rejection sampler).

    The cos-angle-to-mean marginal w is sampled by rejection from a Beta
    envelope; tangent directions are uniform in the hyperplane orthogonal to
    mu. Deterministic given `rng`. kappa <= 0 degenerates to the uniform
    distribution on the sphere.
    """
    d = mu.shape[0]
    if n == 0:
        return np.empty((0, d))
    if kappa <= 0:
        x = rng.standard_normal((n, d))
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), _EPS)

    b = (d - 1) / (2.0 * kappa + np.sqrt(4.0 * kappa**2 + (d - 1) ** 2))
    x0 = (1.0 - b) / (1.0 + b)
    c = kappa * x0 + (d - 1) * np.log(1.0 - x0 * x0)

    ws: list[float] = []
    while len(ws) < n:
        m = max(n - len(ws), 16)
        z = rng.beta((d - 1) / 2.0, (d - 1) / 2.0, size=m)
        w = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
        u = rng.uniform(size=m)
        accept = kappa * w + (d - 1) * np.log1p(-x0 * w) - c >= np.log(u)
        ws.extend(w[accept].tolist())
    w = np.asarray(ws[:n])

    v = rng.standard_normal((n, d))
    v -= np.outer(v @ mu, mu)  # project onto the tangent hyperplane of mu
    v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), _EPS)
    x = w[:, None] * mu[None, :] + np.sqrt(np.maximum(1.0 - w * w, 0.0))[:, None] * v
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), _EPS)


class VMFWorld:
    """Analytic vMF class universe. See module docstring for the full contract."""

    def __init__(
        self,
        seed: int,
        k_known: int = 8,
        k_novel: int = 3,
        n_burst: int = 0,
        d: int = 32,
        separation_deg: float = 75.0,
        kappa_known: float | Sequence[float] = 150.0,
        kappa_novel: float | Sequence[float] = 150.0,
        kappa_burst: float | Sequence[float] = 500.0,
    ) -> None:
        if not 0.0 < separation_deg <= 90.0:
            raise ValueError("separation_deg must be in (0, 90] for the equiangular construction")
        k_total = k_known + k_novel + n_burst
        if k_total > d - 1:
            raise ValueError(f"need k_known+k_novel+n_burst <= d-1, got {k_total} > {d - 1}")

        self.seed = int(seed)
        self.d = int(d)
        self.separation_deg = float(separation_deg)
        self.known_names = [f"known_{i:02d}" for i in range(k_known)]
        self.novel_names = [f"novel_{i:02d}" for i in range(k_novel)]
        self.burst_names = [f"burst_{i:02d}" for i in range(n_burst)]
        self.class_names = self.known_names + self.novel_names + self.burst_names

        kappas = (
            _kappa_list(kappa_known, k_known, "kappa_known")
            + _kappa_list(kappa_novel, k_novel, "kappa_novel")
            + _kappa_list(kappa_burst, n_burst, "kappa_burst")
        )
        self._kappas = dict(zip(self.class_names, kappas))
        self._means = dict(zip(self.class_names, self._build_means(k_total)))

    # ------------------------------------------------------------------ means
    def _build_means(self, k: int) -> np.ndarray:
        """Equiangular unit means: pairwise dot exactly cos(separation_deg)."""
        rng = make_rng(self.seed, "vmf_world/means")
        g = rng.standard_normal((self.d, k + 1))
        q, _ = np.linalg.qr(g)
        # Fix LAPACK's per-column sign ambiguity deterministically.
        signs = np.sign(q[np.argmax(np.abs(q), axis=0), np.arange(k + 1)])
        q = q * signs[None, :]
        u, basis = q[:, 0], q[:, 1:]
        c = np.cos(np.radians(self.separation_deg))
        means = np.sqrt(c) * u[:, None] + np.sqrt(1.0 - c) * basis
        return (means / np.linalg.norm(means, axis=0, keepdims=True)).T  # (k, d)

    # -------------------------------------------------------------- analytics
    def true_mean(self, name: str) -> np.ndarray:
        """Exact vMF mean direction of `name` (copy; (D,) unit vector)."""
        return self._means[name].copy()

    def true_mean_class_names(self) -> list[str]:
        return list(self.class_names)

    def true_means(self) -> np.ndarray:
        """(K, D) true mean directions, rows aligned with true_mean_class_names()."""
        return np.stack([self._means[n] for n in self.class_names])

    def kappa(self, name: str) -> float:
        """True concentration parameter of class `name`."""
        return self._kappas[name]

    def true_pairwise_angles_deg(self) -> np.ndarray:
        """(K, K) matrix of exact pairwise angles between class means, degrees."""
        m = self.true_means()
        return np.degrees(np.arccos(np.clip(m @ m.T, -1.0, 1.0)))

    # --------------------------------------------------------------- sampling
    def sample_class(self, name: str, n: int, stream: str = "adhoc") -> np.ndarray:
        """(n, D) fresh vMF samples of class `name` under substream `stream`.

        Pure function of (seed, stream, name, n): identical calls return
        identical arrays; distinct `stream` labels give independent samples.
        """
        rng = make_rng(self.seed, f"vmf_world/{stream}/{name}")
        return sample_vmf(self._means[name], self._kappas[name], n, rng)

    def distractor_point(self, i: int) -> np.ndarray:
        """(D,) one-off outlier: a uniformly random unit vector, deterministic
        per index. In d>=32 dims it lands ~90 degrees from every class mean
        with overwhelming probability (isolated by construction)."""
        rng = make_rng(self.seed, f"vmf_world/distractor_{i:02d}")
        v = rng.standard_normal(self.d)
        return v / max(float(np.linalg.norm(v)), _EPS)

    # ------------------------------------------------------------------ pools
    def make_pool(self, names: Sequence[str], n_per_class: int, stream: str) -> FixturePool:
        """Concatenated per-class samples (class-blocked, in `names` order)."""
        xs = [self.sample_class(name, n_per_class, stream=stream) for name in names]
        labels = np.array([name for name in names for _ in range(n_per_class)])
        return FixturePool(x=np.vstack(xs) if xs else np.empty((0, self.d)), labels=labels)

    def t0_pool(self, n_per_class: int = 200) -> FixturePool:
        """The "T0 training split": known classes only (LTM-init material)."""
        return self.make_pool(self.known_names, n_per_class, stream="t0")

    def ind_test_pool(self, n_per_class: int = 50) -> FixturePool:
        """Held-out known-class samples, independent of t0_pool."""
        return self.make_pool(self.known_names, n_per_class, stream="ind_test")

    def novel_pool(self, n_per_class: int = 100) -> FixturePool:
        """Samples of the novel classes (ground-truth-labeled)."""
        return self.make_pool(self.novel_names, n_per_class, stream="novel")

    # ----------------------------------------------------------------- stream
    def make_stream(self, schedule: Sequence[Segment]) -> Stream:
        """Deterministic interleaved stream from a phase schedule.

        Per segment i: fresh samples under substream "stream/seg{i:03d}" (so a
        class recurring across segments contributes independent draws), plus
        the segment's one-off distractors; shuffled with its own substream when
        segment.shuffle, else laid down contiguously.
        """
        xs: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        seg_ids: list[np.ndarray] = []
        for i, seg in enumerate(schedule):
            seg_x: list[np.ndarray] = []
            seg_labels: list[str] = []
            for name in sorted(seg.counts):
                n = int(seg.counts[name])
                seg_x.append(self.sample_class(name, n, stream=f"stream/seg{i:03d}"))
                seg_labels.extend([name] * n)
            for j in seg.distractors:
                seg_x.append(self.distractor_point(j)[None, :])
                seg_labels.append(f"distractor_{j:02d}")
            x = np.vstack(seg_x) if seg_x else np.empty((0, self.d))
            lab = np.array(seg_labels)
            if seg.shuffle and len(lab) > 1:
                perm = make_rng(self.seed, f"vmf_world/stream/shuffle{i:03d}").permutation(len(lab))
                x, lab = x[perm], lab[perm]
            xs.append(x)
            labels.append(lab)
            seg_ids.append(np.full(len(lab), i, dtype=np.int64))
        return Stream(
            x=np.vstack(xs) if xs else np.empty((0, self.d)),
            labels=np.concatenate(labels) if labels else np.empty(0, dtype="<U1"),
            segment_ids=np.concatenate(seg_ids) if seg_ids else np.empty(0, dtype=np.int64),
        )
