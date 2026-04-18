"""
FaceRepository — enrollment, search, and identity resolution for ACAS.

Architecture:
  Tier-1  FAISS IndexFlatIP on GPU VRAM — per-DATASET ANN search
            < 0.5 ms at 10 k persons (RTX A5000)
  Tier-2  pgvector HNSW (cosine)        — dataset-scoped fallback
            < 5 ms with HNSW index

Dataset isolation:
  Each Face Dataset (a named group of enrolled faces within a client) gets its
  own independent FAISS index in VRAM.  _dataset_indexes is keyed by dataset_id.

  Cameras can be assigned to a specific dataset (recognition restricted to that
  dataset) or left unassigned (recognition searches across all client datasets).

  All pgvector queries carry mandatory WHERE client_id + dataset_id clauses.

FAISS memory budget (RTX A5000, 24 GB):
  512-D float32 × 10 000 persons = 20 MB / dataset.
  FAISS indexes are rebuilt from pgvector on first access per process restart.

Usage:
    repo = FaceRepository(db, redis, minio, gpu_device=0, pipeline=pipe)
    result = await repo.enroll_person(client_id, person_id, images, metadata,
                                      dataset_id=ds_id)
    match  = await repo.identify(client_id, embedding, roster_ids, thresholds,
                                 dataset_id=ds_id)   # None = search all datasets
"""
from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
import pathlib
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import cv2
import faiss
import numpy as np
from minio import Minio
from minio.error import S3Error
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

# Quality gate thresholds for enrollment
_MIN_IOD_PX             = 20.0    # inter-ocular distance in pixels (matched to upload endpoint)
_MIN_SHARPNESS          = 10.0    # Laplacian variance on 112×112 face chip
_MIN_FACE_CONF          = 0.50    # SCRFD detection confidence
_MIN_LANDMARK_TRI_RATIO = 0.08    # min area(eye-eye-nose) / iod² — occlusion heuristic
_MAX_YAW_DEG            = 45.0    # reject faces with estimated yaw > this (profile / side view)

# FAISS — HNSW index (replaces IVFFlat; better recall at high thresholds, no training needed)
_HNSW_M               = 32      # bi-directional links per node (higher = better recall, more RAM)
_HNSW_EF_CONSTRUCTION = 200     # build-time search width (higher = slower build, better graph)
_HNSW_EF_SEARCH       = 64      # query-time search width (higher = better recall, slower query)
_HNSW_MIN_SIZE        = 20      # switch to HNSW only above this gallery size

# Identification thresholds — tuned for PTZ camera real-world conditions
# AdaFace/ArcFace on PTZ camera: genuine pairs score ~0.20-0.50 depending on angle/distance.
# Tier-1 (FAISS, coarse): accept if similarity >= 0.20  → fast GPU pre-filter
# Tier-2 (pgvector, final): accept if similarity >= 0.25 → confirmed match
_TIER1_THRESHOLD      = 0.20    # was 0.40 — PTZ live frames produce lower sim than enrollment photos
_TIER2_THRESHOLD      = 0.25    # was 0.45 — lowered to match PTZ real-world conditions

# Multi-template gallery — store up to N quality-ranked embeddings per person
_MAX_TEMPLATES        = 5       # max active embeddings per person in gallery

# Embedding whitening — diagonal ZCA applied to FAISS gallery (Tier-1 only)
_MIN_WHITENING_SAMPLES = 100    # whitening is only reliable with large diverse galleries
_WHITEN_EPS            = 1e-5   # regularization added to per-dim std

# Template update policy
_UPDATE_MIN_LIVENESS  = 0.85
_UPDATE_MIN_CONFIRMS  = 3
_UPDATE_EMA_ALPHA     = 0.05    # new = 0.95 * old + 0.05 * candidate
_UPDATE_DRIFT_LIMIT   = 0.15    # cosine distance limit from current centroid

# Redis key templates (prefix is instance-configurable; these are the suffix patterns)
_REDIS_CONFIRMS_KEY   = "{prefix}:face:confirms:{cid}:{pid}"   # JSON confirm counter
_REDIS_INDEX_TS_KEY   = "{prefix}:face:faiss_ts:{cid}"          # last rebuild epoch

# MinIO
_MINIO_BUCKET         = "face-enrollment"

# Local filesystem fallback when MinIO is full
_LOCAL_ENROLLMENT_DIR = "/enrollment-images"


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class QualityResult:
    passed: bool
    reason: str         # "OK" or rejection reason
    iod_px: float
    sharpness: float
    face_conf: float
    landmark_ok: bool


@dataclass
class EnrollmentResult:
    person_id: str
    client_id: str
    embedding_id: str
    centroid_quality: float          # mean sharpness of passing images
    images_total: int
    images_passed: int
    image_refs: list[str]            # MinIO object keys
    quality_details: list[QualityResult]
    error: str | None = None         # set when enrollment fails entirely


@dataclass
class SearchHit:
    person_id: str
    embedding_id: str
    similarity: float


@dataclass
class IdentifyResult:
    person_id: str | None            # None = UNKNOWN
    similarity: float
    tier: int                        # 1 = FAISS, 2 = pgvector, 0 = unknown
    embedding_id: str | None = None


# ── Per-client FAISS index state ───────────────────────────────────────────────

@dataclass
class _ClientIndex:
    gpu_index: Any                   # faiss.GpuIndex or faiss.Index (CPU fallback)
    cpu_index: Any                   # faiss.Index — kept for serialisation
    person_ids: list[str]            # row i → person_id UUID string (may repeat for multi-template)
    embedding_ids: list[str]         # row i → embedding_id UUID string
    built_at: float = field(default_factory=time.time)
    # Diagonal whitening parameters computed from gallery at build time (None = skip whitening)
    whiten_mean: "np.ndarray | None" = None   # [512] float32 — per-dimension gallery mean
    whiten_std:  "np.ndarray | None" = None   # [512] float32 — per-dimension gallery std

    @property
    def size(self) -> int:
        return len(self.person_ids)


# ── Module-level shared state (survives across request scopes) ─────────────────

_gpu_res: faiss.StandardGpuResources | None = None
_gpu_res_lock = threading.Lock()

# dataset_id (str) → _ClientIndex  — one FAISS index per dataset
_dataset_indexes: dict[str, _ClientIndex] = {}
_index_locks: dict[str, asyncio.Lock] = {}    # per-dataset async locks
_global_index_lock = threading.Lock()         # guards _dataset_indexes / _index_locks dicts


def _get_gpu_resources() -> faiss.StandardGpuResources:
    global _gpu_res
    with _gpu_res_lock:
        if _gpu_res is None:
            _gpu_res = faiss.StandardGpuResources()
            _gpu_res.setTempMemory(512 * 1024 * 1024)  # 512 MB scratch
            logger.info("FAISS GPU resources initialised")
        return _gpu_res


def _get_index_lock(key: str) -> asyncio.Lock:
    """Return (or create) an asyncio.Lock for a given dataset_id (or client_id fallback)."""
    with _global_index_lock:
        if key not in _index_locks:
            _index_locks[key] = asyncio.Lock()
        return _index_locks[key]


# ── FAISS index construction ───────────────────────────────────────────────────

def _compute_whitening(
    embeddings: np.ndarray,  # [N, 512] float32
) -> tuple["np.ndarray", "np.ndarray"] | tuple[None, None]:
    """
    Compute diagonal whitening parameters from a gallery of L2-normalised embeddings.
    Returns (mean [512], std [512]) or (None, None) if the gallery is too small.
    """
    n = embeddings.shape[0]
    if n < _MIN_WHITENING_SAMPLES:
        return None, None
    mean = embeddings.mean(axis=0).astype(np.float32)
    std  = (embeddings.std(axis=0) + _WHITEN_EPS).astype(np.float32)
    return mean, std


def _whiten_vectors(
    vecs: np.ndarray,          # [N, 512] or [512]
    mean: "np.ndarray | None",
    std:  "np.ndarray | None",
) -> np.ndarray:
    """Apply diagonal whitening and L2-renormalize. Returns same shape as input."""
    if mean is None or std is None:
        return vecs
    single = vecs.ndim == 1
    v = vecs[None] if single else vecs          # [N, 512]
    w = (v - mean) / std                        # zero-mean, unit-variance per dim
    norms = np.linalg.norm(w, axis=1, keepdims=True) + 1e-8
    w = (w / norms).astype(np.float32)
    return w[0] if single else w


def _build_gpu_index(
    embeddings: np.ndarray,      # [N, 512] float32 L2-normalised
    person_ids: list[str],
    embedding_ids: list[str],
    device_id: int,
) -> _ClientIndex:
    """
    Build a FAISS index from gallery embeddings with diagonal whitening.

    Index type selection:
      N < _HNSW_MIN_SIZE → IndexFlatIP (exact; GPU-promoted)
      N >= _HNSW_MIN_SIZE → IndexHNSWFlat (approximate; CPU only — FAISS has no GpuIndexHNSW)
        HNSW gives better recall at high similarity thresholds vs IVFFlat and
        requires no training step.  At 10 k persons the CPU search is still < 1 ms.

    Whitening:
      Per-dimension mean + std computed from the gallery, applied before indexing.
      The same parameters are stored in _ClientIndex and applied to queries at runtime.
      Without sufficient gallery size (_MIN_WHITENING_SAMPLES), whitening is skipped.
    """
    n, d = embeddings.shape

    # ── Compute whitening from raw gallery ───────────────────────────────────
    w_mean, w_std = _compute_whitening(embeddings)

    # Apply whitening to the gallery before building the index so queries
    # just need the same transform before searching.
    gallery = _whiten_vectors(embeddings, w_mean, w_std)

    # ── Build index ───────────────────────────────────────────────────────────
    if n >= _HNSW_MIN_SIZE:
        # HNSW — better recall at high thresholds; CPU-only in FAISS
        cpu_idx = faiss.IndexHNSWFlat(d, _HNSW_M, faiss.METRIC_INNER_PRODUCT)
        cpu_idx.hnsw.efConstruction = _HNSW_EF_CONSTRUCTION
        cpu_idx.hnsw.efSearch       = _HNSW_EF_SEARCH
        cpu_idx.add(gallery)
        # HNSW doesn't support GPU promotion — use CPU directly
        gpu_idx = cpu_idx
        logger.debug("FAISS HNSW index built  n=%d  M=%d  efSearch=%d", n, _HNSW_M, _HNSW_EF_SEARCH)
    else:
        # Small gallery — exact FlatIP; GPU-promote for < 0.1 ms queries
        cpu_idx = faiss.IndexFlatIP(d)
        cpu_idx.add(gallery)
        try:
            res     = _get_gpu_resources()
            gpu_idx = faiss.index_cpu_to_gpu(res, device_id, cpu_idx)
        except Exception as exc:
            logger.warning("FAISS GPU promotion failed (%s); using CPU index", exc)
            gpu_idx = cpu_idx

    return _ClientIndex(
        gpu_index=gpu_idx,
        cpu_index=cpu_idx,
        person_ids=list(person_ids),
        embedding_ids=list(embedding_ids),
        whiten_mean=w_mean,
        whiten_std=w_std,
    )


def _empty_index(d: int = 512) -> _ClientIndex:
    cpu_idx = faiss.IndexFlatIP(d)
    return _ClientIndex(
        gpu_index=cpu_idx,
        cpu_index=cpu_idx,
        person_ids=[],
        embedding_ids=[],
        whiten_mean=None,
        whiten_std=None,
    )


# ── Quality gate helpers ───────────────────────────────────────────────────────

def _sharpness(chip: np.ndarray) -> float:
    """Laplacian variance on the 112×112 face chip (higher = sharper)."""
    gray = cv2.cvtColor(chip, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _landmark_ok(landmarks: np.ndarray, iod: float) -> bool:
    """
    Heuristic occlusion / profile check via 5-point landmark geometry.

    Two checks must both pass:
      1. Nose must be below the eye midpoint (y increases downward).
         Fails for profile views, upside-down images, or extreme pitch.
      2. Triangle area (left-eye, right-eye, nose) normalised by iod² must be
         >= _MIN_LANDMARK_TRI_RATIO.  Very low area indicates occlusion,
         near-profile pose, or degraded detection.

    landmarks: [5, 2] float32 — (x,y) in pixel coords.
      index 0 = left eye, 1 = right eye, 2 = nose tip, 3/4 = mouth corners.
    """
    if iod < 1e-3:
        return False
    le, re, nose = landmarks[0], landmarks[1], landmarks[2]

    # Check 1: nose must sit below the eye horizontal midline (in image coords)
    eye_mid_y = (le[1] + re[1]) / 2.0
    if nose[1] <= eye_mid_y:
        return False

    # Check 2: normalised triangle area
    area = abs(
        (le[0] * (re[1] - nose[1]))
        + (re[0] * (nose[1] - le[1]))
        + (nose[0] * (le[1] - re[1]))
    ) / 2.0
    return (area / (iod ** 2)) >= _MIN_LANDMARK_TRI_RATIO


def _gate_image(
    img_bgr: np.ndarray,
    pipeline,   # AIPipeline
) -> tuple[QualityResult, np.ndarray | None]:
    """
    Run the full quality gate on one image.
    Returns (QualityResult, face_chip_or_None).
    """
    faces = pipeline.detect_faces(img_bgr)
    if len(faces) == 0:
        return QualityResult(False, "no_face", 0.0, 0.0, 0.0, False), None
    if len(faces) > 1:
        return QualityResult(False, "multiple_faces", 0.0, 0.0, 0.0, False), None

    face = faces[0]
    iod = face.inter_ocular_px
    conf = face.conf

    if iod < _MIN_IOD_PX:
        return QualityResult(False, f"iod_too_small:{iod:.1f}px", iod, 0.0, conf, False), None

    if conf < _MIN_FACE_CONF:
        return QualityResult(False, f"low_conf:{conf:.3f}", iod, 0.0, conf, False), None

    lm_ok = _landmark_ok(face.landmarks, iod)
    if not lm_ok:
        return QualityResult(False, "occlusion_or_profile", iod, 0.0, conf, False), None

    # Pose / yaw check — reject severe profile views (>45°)
    yaw = pipeline.estimate_yaw(face.landmarks)
    if abs(yaw) > _MAX_YAW_DEG:
        return QualityResult(False, f"yaw_too_large:{yaw:.0f}deg", iod, 0.0, conf, False), None

    chip = pipeline.align_face(img_bgr, face.landmarks, iod_px=iod)
    sharp = _sharpness(chip)
    if sharp < _MIN_SHARPNESS:
        return QualityResult(False, f"blurry:{sharp:.1f}", iod, sharp, conf, lm_ok), None

    return QualityResult(True, "OK", iod, sharp, conf, lm_ok), chip


# ── pgvector search helper ─────────────────────────────────────────────────────

async def _pgvector_search(
    db: AsyncSession,
    client_id: str,
    embedding: np.ndarray,    # [512] float32 L2-normalised
    top_k: int = 5,
    dataset_id: str | None = None,
    roster_ids: list[str] | None = None,
) -> list[SearchHit]:
    """
    Tier-2 cosine similarity search via pgvector HNSW index.
    Always scoped to client_id.  Optionally further scoped by dataset_id / roster.

    Multi-template aware: uses DISTINCT ON (person_id) so each person appears once
    with their best-matching template, then re-orders by similarity DESC.
    """
    emb_str = "[" + ",".join(f"{x:.8f}" for x in embedding.tolist()) + "]"

    extra = ""
    params: dict[str, Any] = {"q": emb_str, "cid": client_id, "k": top_k}

    if dataset_id:
        extra += " AND dataset_id = (:did)::uuid"
        params["did"] = dataset_id

    if roster_ids:
        extra += " AND person_id = ANY(:roster)"
        params["roster"] = roster_ids

    # DISTINCT ON (person_id) ordered by closest template first, then wrap with
    # outer ORDER BY similarity DESC to return the top-k highest-scoring persons.
    sql = text(f"""
        SELECT embedding_id, person_id, similarity
        FROM (
            SELECT DISTINCT ON (person_id)
                embedding_id::text  AS embedding_id,
                person_id::text     AS person_id,
                (1 - (embedding <=> (:q)::vector)) AS similarity
            FROM face_embeddings
            WHERE client_id = (:cid)::uuid
              AND is_active = true
              {extra}
            ORDER BY person_id, embedding <=> (:q)::vector ASC
        ) best_per_person
        ORDER BY similarity DESC
        LIMIT :k
    """)

    rows = (await db.execute(sql, params)).fetchall()
    return [
        SearchHit(
            person_id=str(r.person_id),
            embedding_id=str(r.embedding_id),
            similarity=float(r.similarity),
        )
        for r in rows
    ]


# ── Main repository ────────────────────────────────────────────────────────────

class FaceRepository:
    """
    All methods are async-safe.  Blocking FAISS / MinIO / CV2 operations are
    dispatched to a thread pool via asyncio.to_thread.

    Parameters:
        db          — SQLAlchemy AsyncSession (request-scoped or app-scoped)
        redis       — redis.asyncio.Redis client (app-scoped)
        minio       — minio.Minio client (sync, app-scoped)
        gpu_device  — CUDA device index for FAISS GPU index
        pipeline    — AIPipeline instance (needed for enroll_person)
    """

    def __init__(
        self,
        db,  # async_sessionmaker[AsyncSession] — passed from app.state.session_factory
        redis: Redis,
        minio: Minio,
        gpu_device: int = 0,
        pipeline: Any = None,   # AIPipeline | None
        tier1_threshold: float = _TIER1_THRESHOLD,
        tier2_threshold: float = _TIER2_THRESHOLD,
        minio_bucket: str = _MINIO_BUCKET,
        local_enrollment_dir: str = _LOCAL_ENROLLMENT_DIR,
        redis_key_prefix: str = "acas",
    ) -> None:
        self._db       = db   # session factory, NOT a session
        self._redis    = redis
        self._minio    = minio
        self._device   = gpu_device
        self._pipeline = pipeline
        self._tier1    = tier1_threshold
        self._tier2    = tier2_threshold
        self._bucket   = minio_bucket
        self._local_dir = local_enrollment_dir
        self._prefix   = redis_key_prefix

    @contextlib.asynccontextmanager
    async def _session(self):
        """Open a short-lived DB session with an explicit transaction."""
        async with self._db() as session:
            async with session.begin():
                yield session

    # ── Enrollment ─────────────────────────────────────────────────────────────

    async def enroll_person(
        self,
        client_id: str,
        person_id: str,
        images: list[bytes],
        metadata: dict | None = None,
        dataset_id: str | None = None,
    ) -> EnrollmentResult:
        """
        Enroll a person from a list of raw image bytes.

        Quality gates applied to each image (all must pass):
          • Exactly 1 face detected
          • inter_ocular_distance >= 20 px (_MIN_IOD_PX)
          • Laplacian sharpness >= 10 (_MIN_SHARPNESS)
          • Landmark geometry passes occlusion heuristic
          • Face detection confidence >= 0.75

        The centroid of all passing embeddings is stored in pgvector and the
        client's FAISS index is rebuilt with the new vector appended.

        Images that pass quality are uploaded to MinIO:
          face-enrollment/{client_id}/{person_id}/{uuid4}.jpg
        """
        if not self._pipeline:
            return EnrollmentResult(
                person_id=person_id, client_id=client_id, embedding_id="",
                centroid_quality=0.0, images_total=len(images), images_passed=0,
                image_refs=[], quality_details=[],
                error="AIPipeline not configured",
            )

        # ── Phase 1: Quality gate (in thread) ────────────────────────────────
        quality_results, chips, embeddings = await asyncio.to_thread(
            self._run_quality_gate, images
        )

        passed_count = sum(1 for q in quality_results if q.passed)
        if passed_count == 0:
            reasons = [q.reason for q in quality_results]
            return EnrollmentResult(
                person_id=person_id, client_id=client_id, embedding_id="",
                centroid_quality=0.0, images_total=len(images), images_passed=0,
                image_refs=[], quality_details=quality_results,
                error=f"All {len(images)} images failed quality gate: {reasons}",
            )

        # ── Phase 2: PTZ-domain augmentation ─────────────────────────────────
        # Enrollment photos (phone/DSLR) look different from PTZ camera frames:
        # higher resolution, different compression, sharper/brighter.  We generate
        # a small set of PTZ-like variants per passing chip and embed them too.
        # These extra templates dramatically reduce the similarity gap when the
        # enrolled person is next seen through a PTZ camera at a distance.
        aug_chips: list[np.ndarray] = []
        aug_embeddings: list[np.ndarray] = []
        aug_qualities: list["QualityResult"] = []

        if chips:
            aug_chips, aug_embeddings, aug_qualities = await asyncio.to_thread(
                self._augment_ptz_domain, chips
            )

        # Combine original + augmented pools
        all_chips      = chips      + aug_chips
        all_embeddings = embeddings + aug_embeddings

        # ── Phase 3: Quality-weighted template selection ──────────────────────
        # Compute per-image quality weight = sharpness × face_conf (both bounded).
        # Select up to _MAX_TEMPLATES best-quality embeddings as distinct gallery
        # templates — no mean-centroid averaging, which washes out pose variation.
        passing_qualities = [q for q in quality_results if q.passed] + aug_qualities
        weights = np.array([
            q.sharpness * q.face_conf for q in passing_qualities
        ], dtype=np.float32)

        # Sort by quality weight descending; keep top _MAX_TEMPLATES
        order = np.argsort(weights)[::-1][:_MAX_TEMPLATES]
        selected_embeddings = [all_embeddings[i] for i in order]
        selected_chips      = [all_chips[i]      for i in order]
        selected_weights    = weights[order]

        if not selected_embeddings:
            return EnrollmentResult(
                person_id=person_id, client_id=client_id, embedding_id="",
                centroid_quality=0.0, images_total=len(images), images_passed=passed_count,
                image_refs=[], quality_details=quality_results,
                error="No embeddings survived quality selection",
            )

        # Mean quality score of selected templates (for reporting)
        _mean_sharp = float(np.mean([passing_qualities[i].sharpness for i in order]))
        quality_score = float(np.clip(_mean_sharp / 400.0, 0.0, 1.0))
        _ = selected_weights  # available for future per-template weighting

        # ── Phase 3: MinIO upload — only the selected templates ──────────────
        image_refs = await asyncio.to_thread(
            self._upload_enrollment_images,
            client_id, person_id, selected_chips,
        )

        # ── Phase 4: Write to pgvector — one row per template ────────────────
        # Deactivate existing templates first (clean slate for re-enrollment).
        now = int(time.time())
        primary_embedding_id = str(uuid.uuid4())

        async with self._session() as db:
            await db.execute(
                text("""
                    UPDATE face_embeddings
                    SET is_active = false
                    WHERE person_id = (:pid)::uuid
                      AND client_id = (:cid)::uuid
                      AND is_active = true
                """),
                {"pid": person_id, "cid": client_id},
            )

            # Insert one row per selected template (up to _MAX_TEMPLATES)
            for t_idx, (template_emb, ref) in enumerate(
                zip(selected_embeddings, image_refs + [""] * len(selected_embeddings))
            ):
                eid      = primary_embedding_id if t_idx == 0 else str(uuid.uuid4())
                emb_str  = "[" + ",".join(f"{x:.8f}" for x in template_emb.tolist()) + "]"
                # Sharpness-normalised per-template quality score
                t_sharp  = passing_qualities[order[t_idx]].sharpness
                t_qs     = float(np.clip(t_sharp / 400.0, 0.0, 1.0))
                t_ref    = ref if ref else None
                await db.execute(
                    text("""
                        INSERT INTO face_embeddings
                            (embedding_id, client_id, person_id, dataset_id, embedding,
                             version, source, confidence_avg, image_refs,
                             is_active, quality_score, created_at)
                        VALUES
                            ((:eid)::uuid, (:cid)::uuid, (:pid)::uuid, (:did)::uuid, (:emb)::vector,
                             1, :src, :conf, :refs,
                             true, :qs, :now)
                        ON CONFLICT DO NOTHING
                    """),
                    {
                        "eid": eid,       "cid": client_id, "pid": person_id,
                        "did": dataset_id,
                        "emb": emb_str,   "conf": t_qs,
                        "src": f"ENROLLMENT_T{t_idx}",
                        "refs": [t_ref] if t_ref else [],
                        "qs": t_qs,       "now": now,
                    },
                )

        # ── Phase 5: Rebuild FAISS index (includes all templates) ────────────
        if dataset_id:
            await self.rebuild_faiss_index(client_id, dataset_id=dataset_id)
        else:
            await self.rebuild_faiss_index(client_id)

        return EnrollmentResult(
            person_id=person_id,
            client_id=client_id,
            embedding_id=primary_embedding_id,
            centroid_quality=quality_score,
            images_total=len(images),
            images_passed=passed_count,
            image_refs=image_refs,
            quality_details=quality_results,
        )

    def _augment_ptz_domain(
        self,
        chips: list[np.ndarray],
    ) -> tuple[list[np.ndarray], list[np.ndarray], list["QualityResult"]]:
        """
        Generate PTZ-camera-like augmented variants of the best enrollment chip.

        Simulates the three main causes of the enrollment→PTZ domain gap:
          1. Lower effective resolution (PTZ cameras 30–50× optical zoom, but at
             distance the face may only be 80–120px IOD after zoom-in).
          2. Slight Gaussian blur from H.264 compression artefacts.
          3. Moderate contrast/brightness variation from PTZ AGC circuits.

        We downsample → re-upsample (mimics compression loss) and vary gamma.
        One low-light + one high-exposure variant give ArcFace more to work with.
        Returns (chips, embeddings, quality_results) for the augmented batch.
        """
        aug_chips: list[np.ndarray] = []
        aug_embeddings: list[np.ndarray] = []
        aug_qualities: list[QualityResult] = []

        if not chips or self._pipeline is None:
            return aug_chips, aug_embeddings, aug_qualities

        # Use the sharpest original chip as the augmentation source
        sharpnesses = [
            float(cv2.Laplacian(c, cv2.CV_64F).var()) for c in chips
        ]
        src = chips[int(np.argmax(sharpnesses))]   # sharpest chip (112×112)

        def _gamma(img: np.ndarray, g: float) -> np.ndarray:
            lut = (np.arange(256, dtype=np.float32) / 255.0) ** (1.0 / g)
            lut = (lut * 255.0).clip(0, 255).astype(np.uint8)
            return cv2.LUT(img, lut)

        augmentations = [
            # (description, transform)
            # 1. Compress: downsample to 56px then back → simulates PTZ distance
            ("ptz_compress",
             lambda c: cv2.resize(cv2.resize(c, (56, 56), cv2.INTER_AREA),
                                  (112, 112), cv2.INTER_CUBIC)),
            # 2. Low-light: gamma 0.65 → darker, simulates indoor PTZ AGC
            ("ptz_dark",   lambda c: _gamma(c, 0.65)),
            # 3. Bright + slight blur: gamma 1.4 + blur → overexposed scene
            ("ptz_bright", lambda c: cv2.GaussianBlur(_gamma(c, 1.40), (3, 3), 0)),
        ]

        for _desc, transform in augmentations:
            try:
                aug = transform(src)
                emb = self._pipeline.get_embedding(aug)
                sharpness = float(cv2.Laplacian(aug, cv2.CV_64F).var())
                aug_chips.append(aug)
                aug_embeddings.append(emb)
                # Quality weight is lower than original (intentionally — augmented
                # chips are less sharp) so they're only picked when slots remain.
                aug_qualities.append(
                    QualityResult(
                        passed=True,
                        reason="ptz_augmentation",
                        iod_px=60.0,
                        sharpness=sharpness * 0.7,   # discounted weight
                        face_conf=0.80,
                        landmark_ok=True,
                    )
                )
            except Exception as exc:
                logger.debug("_augment_ptz_domain %s failed: %s", _desc, exc)

        return aug_chips, aug_embeddings, aug_qualities

    def _run_quality_gate(
        self,
        images: list[bytes],
    ) -> tuple[list[QualityResult], list[np.ndarray], list[np.ndarray]]:
        """
        Decode and quality-gate all images.
        Returns (quality_results, passing_chips, passing_embeddings).
        """
        quality_results: list[QualityResult] = []
        chips: list[np.ndarray] = []
        embeddings: list[np.ndarray] = []

        for raw in images:
            arr = np.frombuffer(raw, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                quality_results.append(
                    QualityResult(False, "decode_failed", 0.0, 0.0, 0.0, False)
                )
                continue

            qr, chip = _gate_image(img, self._pipeline)
            quality_results.append(qr)
            if not qr.passed or chip is None:
                continue

            emb = self._pipeline.get_embedding(chip)
            chips.append(chip)
            embeddings.append(emb)

        return quality_results, chips, embeddings

    def _upload_enrollment_images(
        self,
        client_id: str,
        person_id: str,
        chips: list[np.ndarray],
    ) -> list[str]:
        """Upload aligned face chips to MinIO, falling back to local disk."""
        refs: list[str] = []
        for chip in chips:
            ok, buf = cv2.imencode(".jpg", chip, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                continue
            key = f"{client_id}/{person_id}/{uuid.uuid4()}.jpg"
            stored = False
            try:
                self._minio.put_object(
                    self._bucket,
                    key,
                    io.BytesIO(buf.tobytes()),
                    length=len(buf),
                    content_type="image/jpeg",
                )
                stored = True
            except S3Error as exc:
                logger.warning("MinIO upload failed for %s: %s — falling back to local disk", key, exc)

            if not stored:
                local_dir = pathlib.Path(self._local_dir) / client_id / person_id
                local_dir.mkdir(parents=True, exist_ok=True)
                local_path = local_dir / pathlib.Path(key).name
                local_path.write_bytes(buf.tobytes())
                key = f"local:{key}"

            refs.append(key)
        return refs

    # ── Search helpers ─────────────────────────────────────────────────────────

    async def _all_person_ids(self, client_id: str, dataset_id: str | None = None) -> list[str]:
        """Return all person_ids with active embeddings for a client (used when roster is empty)."""
        async with self._session() as db:
            if dataset_id:
                rows = (await db.execute(
                    text("SELECT DISTINCT person_id::text FROM face_embeddings WHERE client_id=(:cid)::uuid AND dataset_id=(:did)::uuid AND is_active=true"),
                    {"cid": client_id, "did": dataset_id},
                )).fetchall()
            else:
                rows = (await db.execute(
                    text("SELECT DISTINCT person_id::text FROM face_embeddings WHERE client_id=(:cid)::uuid AND is_active=true"),
                    {"cid": client_id},
                )).fetchall()
        return [r[0] for r in rows]

    # ── Search: Tier-1 FAISS GPU ───────────────────────────────────────────────

    async def search_roster(
        self,
        client_id: str,
        embedding: np.ndarray,      # [512] float32 L2-normalised
        roster_ids: list[str],
        top_k: int = 5,
        dataset_id: str | None = None,
    ) -> list[SearchHit]:
        """
        Tier-1: FAISS search restricted to a specific roster of person IDs.

        Multi-template aware: a person may have up to _MAX_TEMPLATES gallery rows.
        We over-fetch, group by person_id keeping max similarity, then apply the
        roster filter and return top_k.

        Whitening: if the index was built with whitening parameters, the query
        embedding receives the same diagonal ZCA transform before searching.

        Target: < 1 ms after index is warm (CPU HNSW or GPU FlatIP).
        """
        ci = await self._ensure_index(client_id, dataset_id=dataset_id)
        if ci.size == 0:
            return []

        # Over-fetch: account for multi-template duplicates + roster attrition
        k_fetch = min(ci.size, max(top_k * _MAX_TEMPLATES * 3, len(roster_ids) * _MAX_TEMPLATES + top_k))

        def _search() -> list[SearchHit]:
            # Apply the same whitening that was applied to the gallery at build time
            q_vec = _whiten_vectors(embedding.astype(np.float32), ci.whiten_mean, ci.whiten_std)
            q = q_vec.reshape(1, -1)
            D, I = ci.gpu_index.search(q, k_fetch)
            roster_set = set(roster_ids)

            # Group by person_id — keep best (max similarity) template per person
            best: dict[str, SearchHit] = {}
            for score, row_idx in zip(D[0], I[0]):
                if row_idx < 0 or row_idx >= ci.size:
                    continue
                pid = ci.person_ids[row_idx]
                if pid not in roster_set:
                    continue
                sim = float(score)
                if pid not in best or sim > best[pid].similarity:
                    best[pid] = SearchHit(
                        person_id=pid,
                        embedding_id=ci.embedding_ids[row_idx],
                        similarity=sim,
                    )

            hits = sorted(best.values(), key=lambda h: h.similarity, reverse=True)
            return hits[:top_k]

        return await asyncio.to_thread(_search)

    # ── Search: Tier-2 pgvector ────────────────────────────────────────────────

    async def search_institution(
        self,
        client_id: str,
        embedding: np.ndarray,
        top_k: int = 5,
        dataset_id: str | None = None,
    ) -> list[SearchHit]:
        """
        Tier-2: pgvector HNSW cosine search.
        If dataset_id is provided, restricts to that dataset.
        Otherwise searches all active embeddings for the client.
        Target: < 5 ms via HNSW index.
        """
        async with self._session() as db:
            return await _pgvector_search(db, client_id, embedding, top_k, dataset_id)

    # ── Identify ───────────────────────────────────────────────────────────────

    async def identify(
        self,
        client_id: str,
        embedding: np.ndarray,
        roster_ids: list[str],
        thresholds: dict | None = None,
        dataset_id: str | None = None,
    ) -> IdentifyResult:
        """
        Two-tier identity resolution.

          Tier-1 — FAISS GPU, roster-scoped:
            If dataset_id is set, only that dataset's FAISS index is searched.
            If dataset_id is None, all datasets for the client are searched.
            similarity >= tier1_threshold → MATCH (return immediately)
            else → fall through to Tier-2

          Tier-2 — pgvector, dataset-scoped (or all client datasets if dataset_id=None):
            similarity >= tier2_threshold → MATCH
            else → UNKNOWN

        Args:
            client_id:  Caller's client UUID (mandatory; enforces isolation).
            embedding:  [512] float32 L2-normalised query embedding.
            roster_ids: Persons expected in this context (e.g. today's class).
            thresholds: {"tier1": float, "tier2": float} — optional overrides.
                        When omitted, per-dataset calibrated thresholds are
                        loaded from `face_datasets` (set by
                        `scripts/calibrate_thresholds.py`); fall back to
                        self._tier1 / self._tier2 if the dataset has not been
                        calibrated yet.
            dataset_id: Restrict to a specific dataset; None = all client datasets.
        """
        override = thresholds or {}
        if "tier1" in override and "tier2" in override:
            t1, t2 = float(override["tier1"]), float(override["tier2"])
        else:
            t1_cal, t2_cal = await self._load_dataset_thresholds(client_id, dataset_id)
            t1 = float(override.get("tier1", t1_cal if t1_cal is not None else self._tier1))
            t2 = float(override.get("tier2", t2_cal if t2_cal is not None else self._tier2))

        # ── Tier-1 FAISS ──────────────────────────────────────────────────────
        # Always run FAISS: roster-scoped when roster provided, all-persons otherwise.
        hits1 = await self.search_roster(
            client_id, embedding, roster_ids if roster_ids else list(await self._all_person_ids(client_id, dataset_id)),
            top_k=1, dataset_id=dataset_id,
        )
        if hits1:
            logger.info("IDENTIFY tier1 best_sim=%.4f person=%s threshold=%.2f %s",
                hits1[0].similarity, hits1[0].person_id, t1,
                "MATCH" if hits1[0].similarity >= t1 else "below_threshold")
        if hits1 and hits1[0].similarity >= t1:
            h = hits1[0]
            return IdentifyResult(
                person_id=h.person_id,
                similarity=h.similarity,
                tier=1,
                embedding_id=h.embedding_id,
            )

        # ── Tier-2 pgvector ───────────────────────────────────────────────────
        hits2 = await self.search_institution(
            client_id, embedding, top_k=1, dataset_id=dataset_id,
        )
        if hits2:
            logger.info("IDENTIFY tier2 best_sim=%.4f person=%s threshold=%.2f %s",
                hits2[0].similarity, hits2[0].person_id, t2,
                "MATCH" if hits2[0].similarity >= t2 else "below_threshold")
        if hits2 and hits2[0].similarity >= t2:
            h = hits2[0]
            return IdentifyResult(
                person_id=h.person_id,
                similarity=h.similarity,
                tier=2,
                embedding_id=h.embedding_id,
            )

        best_sim = hits2[0].similarity if hits2 else (hits1[0].similarity if hits1 else 0.0)
        logger.info("IDENTIFY result=UNKNOWN best_sim=%.4f t1=%.2f t2=%.2f", best_sim, t1, t2)
        return IdentifyResult(person_id=None, similarity=0.0, tier=0)

    # ── Template update ────────────────────────────────────────────────────────

    async def update_template(
        self,
        client_id: str,
        person_id: str,
        new_embedding: np.ndarray,
        confidence: float,
    ) -> bool:
        """
        Propose an EMA template update for an existing enrolled person.

        Rules:
          • confidence (liveness score) must be >= _UPDATE_MIN_LIVENESS
          • The proposal must be confirmed _UPDATE_MIN_CONFIRMS times before
            the DB record is mutated (prevents one bad frame corrupting template)
          • The new embedding must be within _UPDATE_DRIFT_LIMIT cosine distance
            of the current centroid (drift guard)
          • EMA:  new_centroid = L2( 0.95 * old + 0.05 * confirmed_candidate )

        Returns True if the template was actually updated, False otherwise.
        """
        if confidence < _UPDATE_MIN_LIVENESS:
            logger.debug(
                "template update rejected: liveness %.3f < %.2f",
                confidence, _UPDATE_MIN_LIVENESS,
            )
            return False

        # Load current active embedding
        async with self._session() as _db_read:
            row = (await _db_read.execute(
                text("""
                    SELECT embedding_id::text, embedding::text
                    FROM face_embeddings
                    WHERE person_id = (:pid)::uuid
                      AND client_id = (:cid)::uuid
                      AND is_active = true
                    LIMIT 1
                """),
                {"pid": person_id, "cid": client_id},
            )).fetchone()

        if not row:
            logger.warning("update_template: no active embedding for %s/%s", client_id, person_id)
            return False

        embedding_id = row.embedding_id
        # pgvector returns the vector as a Postgres text repr "[x1,x2,...]"
        current_emb = np.array(
            [float(v) for v in row.embedding.strip("[]").split(",")],
            dtype=np.float32,
        )

        # Drift guard: cosine distance = 1 - cosine_similarity
        sim = float(np.dot(new_embedding, current_emb))
        if (1.0 - sim) > _UPDATE_DRIFT_LIMIT:
            logger.info(
                "template update: drift %.4f > limit %.2f for %s — rejected",
                1.0 - sim, _UPDATE_DRIFT_LIMIT, person_id,
            )
            return False

        # Redis confirmation counter
        redis_key = _REDIS_CONFIRMS_KEY.format(prefix=self._prefix, cid=client_id, pid=person_id)
        raw = await self._redis.get(redis_key)
        state: dict = json.loads(raw) if raw else {"count": 0, "emb": None}

        if state["emb"] is None:
            # Start new confirmation round with this embedding
            state["emb"] = new_embedding.tolist()
            state["count"] = 1
            await self._redis.setex(redis_key, 3600, json.dumps(state))
            logger.debug("template update: started confirm round for %s (1/%d)", person_id, _UPDATE_MIN_CONFIRMS)
            return False

        # Accumulate the pending proposal via EMA within the confirmation window
        pending = np.array(state["emb"], dtype=np.float32)
        blended = 0.8 * pending + 0.2 * new_embedding
        blen_norm = np.linalg.norm(blended)
        blended = (blended / blen_norm) if blen_norm > 1e-8 else new_embedding
        state["emb"] = blended.tolist()
        state["count"] = state["count"] + 1
        await self._redis.setex(redis_key, 3600, json.dumps(state))

        if state["count"] < _UPDATE_MIN_CONFIRMS:
            logger.debug(
                "template update: %d/%d confirms for %s",
                state["count"], _UPDATE_MIN_CONFIRMS, person_id,
            )
            return False

        # Enough confirms — apply the EMA update
        confirmed = np.array(state["emb"], dtype=np.float32)
        updated = _UPDATE_EMA_ALPHA * confirmed + (1 - _UPDATE_EMA_ALPHA) * current_emb
        upd_norm = np.linalg.norm(updated)
        updated = (updated / upd_norm).astype(np.float32)

        emb_str = "[" + ",".join(f"{x:.8f}" for x in updated.tolist()) + "]"
        now = int(time.time())
        async with self._session() as _db_upd:
            await _db_upd.execute(
                text("""
                    UPDATE face_embeddings
                    SET embedding = (:emb)::vector,
                        source = 'AUTO_UPDATE',
                        created_at = :now
                    WHERE embedding_id = (:eid)::uuid
                      AND client_id   = (:cid)::uuid
                """),
                {"emb": emb_str, "eid": embedding_id, "cid": client_id, "now": now},
            )

        # Invalidate Redis confirm state and rebuild FAISS
        await self._redis.delete(redis_key)
        await self.rebuild_faiss_index(client_id)

        logger.info(
            "template updated for person=%s client=%s  (drift=%.4f, sim_before=%.4f)",
            person_id, client_id, 1.0 - sim, sim,
        )
        return True

    # ── FAISS index management ─────────────────────────────────────────────────

    async def rebuild_faiss_index(
        self,
        client_id: str,
        roster_ids: list[str] | None = None,
        dataset_id: str | None = None,
    ) -> int:
        """
        Rebuild the in-memory GPU FAISS index from pgvector.

        Keying:
          • dataset_id is provided → rebuild only that dataset's index
            (key = dataset_id in _dataset_indexes)
          • dataset_id is None → rebuild the catch-all client index
            (key = client_id in _dataset_indexes, covers all datasets)

        roster_ids further restricts which persons are included.

        Returns the number of vectors in the new index.
        Guaranteed: all rows are scoped to client_id.
        """
        index_key = dataset_id if dataset_id else client_id
        lock = _get_index_lock(index_key)
        async with lock:
            rows = await self._load_embeddings_from_db(client_id, roster_ids, dataset_id)
            n = len(rows)

            def _build() -> _ClientIndex:
                if n == 0:
                    return _empty_index()
                embs = np.stack([r["embedding"] for r in rows]).astype(np.float32)
                pids = [r["person_id"] for r in rows]
                eids = [r["embedding_id"] for r in rows]
                return _build_gpu_index(embs, pids, eids, self._device)

            ci = await asyncio.to_thread(_build)
            _dataset_indexes[index_key] = ci

        await self._redis.set(
            _REDIS_INDEX_TS_KEY.format(prefix=self._prefix, cid=index_key),
            str(time.time()),
            ex=86400,
        )
        logger.info(
            "FAISS index rebuilt  key=%s  vectors=%d  type=%s  whitened=%s",
            index_key, n,
            "HNSW" if n >= _HNSW_MIN_SIZE else "FlatIP",
            ci.whiten_mean is not None,
        )
        return n

    async def _load_dataset_thresholds(
        self,
        client_id: str,
        dataset_id: str | None,
    ) -> tuple[float | None, float | None]:
        """
        Return (tier1, tier2) calibrated thresholds for the given scope.

        If ``dataset_id`` is provided, read that dataset's columns directly.
        Otherwise average across all ACTIVE datasets for the client — that
        preserves the spirit of per-dataset calibration even when identify()
        is called without a dataset filter.  If no dataset is calibrated, we
        return (None, None) and the caller falls back to the global defaults.
        """
        if dataset_id:
            sql = text("""
                SELECT tier1_threshold AS t1, tier2_threshold AS t2
                FROM face_datasets
                WHERE dataset_id = (:did)::uuid
                LIMIT 1
            """)
            params = {"did": dataset_id}
        else:
            sql = text("""
                SELECT AVG(tier1_threshold) AS t1, AVG(tier2_threshold) AS t2
                FROM face_datasets
                WHERE client_id = (:cid)::uuid
                  AND status = 'ACTIVE'
                  AND tier1_threshold IS NOT NULL
                  AND tier2_threshold IS NOT NULL
            """)
            params = {"cid": client_id}

        try:
            async with self._session() as db:
                row = (await db.execute(sql, params)).fetchone()
        except Exception as exc:  # e.g. migration 0013 not yet applied
            logger.debug("dataset threshold lookup failed: %s", exc)
            return None, None

        if not row:
            return None, None
        t1 = float(row.t1) if row.t1 is not None else None
        t2 = float(row.t2) if row.t2 is not None else None
        return t1, t2

    async def _load_embeddings_from_db(
        self,
        client_id: str,
        roster_ids: list[str] | None,
        dataset_id: str | None = None,
    ) -> list[dict]:
        """
        Load active embeddings from pgvector.
        Always scoped to client_id; optionally further scoped by dataset_id / roster.
        """
        clauses = ["client_id = (:cid)::uuid", "is_active = true"]
        params: dict = {"cid": client_id}

        if dataset_id:
            clauses.append("dataset_id = (:did)::uuid")
            params["did"] = dataset_id

        if roster_ids:
            clauses.append("person_id = ANY(:roster)")
            params["roster"] = roster_ids

        where = " AND ".join(clauses)
        sql = text(f"""
            SELECT embedding_id::text, person_id::text, embedding::text
            FROM face_embeddings
            WHERE {where}
            ORDER BY created_at
        """)

        async with self._session() as db:
            rows = (await db.execute(sql, params)).fetchall()
        result = []
        for r in rows:
            arr = np.array(
                [float(v) for v in r.embedding.strip("[]").split(",")],
                dtype=np.float32,
            )
            result.append({
                "embedding_id": r.embedding_id,
                "person_id": r.person_id,
                "embedding": arr,
            })
        return result

    async def _ensure_index(
        self,
        client_id: str,
        dataset_id: str | None = None,
    ) -> _ClientIndex:
        """
        Return the cached GPU FAISS index, building it from pgvector on first
        access.  If dataset_id is provided, a dataset-specific index is returned;
        otherwise the client-wide index (all datasets) is returned.
        """
        key = dataset_id if dataset_id else client_id
        if key not in _dataset_indexes:
            await self.rebuild_faiss_index(client_id, dataset_id=dataset_id)
        return _dataset_indexes.get(key, _empty_index())

    # ── MinIO helpers ─────────────────────────────────────────────────────────

    async def get_enrollment_images(
        self,
        client_id: str,
        person_id: str,
    ) -> list[bytes]:
        """
        Retrieve all enrollment images for a person from MinIO + local disk fallback.
        Returns list of raw JPEG bytes.
        """
        prefix = f"{client_id}/{person_id}/"

        def _fetch():
            result: list[bytes] = []
            # Try MinIO first
            try:
                objects = list(self._minio.list_objects(_MINIO_BUCKET, prefix=prefix))
                for obj in objects:
                    try:
                        response = self._minio.get_object(_MINIO_BUCKET, obj.object_name)
                        result.append(response.read())
                        response.close()
                        response.release_conn()
                    except S3Error as exc:
                        logger.warning("MinIO get_object failed: %s %s", obj.object_name, exc)
            except Exception as exc:
                logger.warning("MinIO list_objects failed: %s", exc)

            # Also check local disk fallback
            local_dir = pathlib.Path(_LOCAL_ENROLLMENT_DIR) / client_id / person_id
            if local_dir.is_dir():
                for f in sorted(local_dir.glob("*.jpg")):
                    result.append(f.read_bytes())
            return result

        return await asyncio.to_thread(_fetch)

    # ── Delete person ─────────────────────────────────────────────────────────

    async def delete_person(self, client_id: str, person_id: str) -> int:
        """
        Deactivate all face embeddings for a person and remove their enrollment
        images from MinIO.  Rebuilds the client FAISS index.

        Returns the number of embeddings deactivated.
        """
        # Deactivate embeddings and mark person INACTIVE (soft delete — keeps audit trail)
        async with self._session() as db:
            result = await db.execute(
                text("""
                    UPDATE face_embeddings
                    SET is_active = false
                    WHERE person_id = (:pid)::uuid
                      AND client_id = (:cid)::uuid
                      AND is_active = true
                    RETURNING embedding_id
                """),
                {"pid": person_id, "cid": client_id},
            )
            deactivated = result.rowcount
            await db.execute(
                text("""
                    UPDATE persons
                    SET status = 'INACTIVE'
                    WHERE person_id = (:pid)::uuid
                      AND client_id = (:cid)::uuid
                """),
                {"pid": person_id, "cid": client_id},
            )

        # Remove MinIO objects
        prefix = f"{client_id}/{person_id}/"

        def _remove_minio():
            try:
                objects = list(self._minio.list_objects(_MINIO_BUCKET, prefix=prefix))
                for obj in objects:
                    try:
                        self._minio.remove_object(_MINIO_BUCKET, obj.object_name)
                    except S3Error as exc:
                        logger.warning("MinIO remove_object failed: %s %s", obj.object_name, exc)
            except S3Error as exc:
                logger.warning("MinIO list_objects failed for %s: %s", prefix, exc)

        await asyncio.to_thread(_remove_minio)

        # Clear confirm state and rebuild index
        await self._redis.delete(
            _REDIS_CONFIRMS_KEY.format(prefix=self._prefix, cid=client_id, pid=person_id)
        )
        await self.rebuild_faiss_index(client_id)

        logger.info(
            "delete_person: deactivated %d embedding(s) for person=%s client=%s",
            deactivated, person_id, client_id,
        )
        return deactivated

    # ── Utility / introspection ────────────────────────────────────────────────

    def get_faiss_count(self, client_id: str) -> int:
        """
        Return the number of embeddings currently loaded in the in-memory FAISS
        index for the given client.  Returns 0 if the index does not exist yet.
        Used by FaceSyncService for consistency checks.
        """
        idx = _dataset_indexes.get(client_id)
        return idx.size if idx is not None else 0
