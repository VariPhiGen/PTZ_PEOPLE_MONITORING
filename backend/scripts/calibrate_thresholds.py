#!/usr/bin/env python3
"""
Adaptive 1:N threshold calibration for ACAS face datasets.

Why this exists
───────────────
The default Tier-1 / Tier-2 thresholds (0.20 / 0.25) were tuned for a generic
PTZ deployment.  Real deployments drift: different cameras, different lighting,
different demographics, and different AdaFace fine-tunes all shift the genuine
and impostor similarity distributions.  A fixed threshold that was safe on one
dataset is either too permissive (higher FAR) or too strict (higher FRR) on
another.

What we compute
───────────────
For a given dataset we split the enrolled embeddings by person_id, then:

  genuine pairs  — every same-person pair (p_i_a · p_i_b) across all persons
                   that have ≥ 2 active embeddings
  impostor pairs — up to IMPOSTOR_SAMPLES random cross-person pairs

Then, given a target False-Accept Rate (FAR), we pick:

  tier2 = smallest threshold s.t. impostor FAR  ≤ target_far       (final match)
  tier1 = tier2 − TIER1_MARGIN, floored at the minimum FAR point   (coarse gate)

Tier-1 is deliberately more permissive: it only advances candidates to the
Tier-2 pgvector check, so false positives cost one extra DB round-trip, not a
wrong identity.  Tier-2 is the authoritative match gate.

We also emit FRR at the chosen Tier-2 operating point and a histogram digest
for the admin UI (percentiles of the two distributions).

Usage
─────
  # calibrate all datasets with ≥ MIN_PERSONS enrolled
  python scripts/calibrate_thresholds.py --all

  # calibrate a single dataset, higher target FAR (1e-3) for a small gallery
  python scripts/calibrate_thresholds.py \\
      --dataset-id <uuid> --target-far 1e-3

  # dry run — show what would be written without updating the DB
  python scripts/calibrate_thresholds.py --all --dry-run

Docker:
  docker exec acas-backend python /app/scripts/calibrate_thresholds.py --all
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from typing import Any

import numpy as np

# Allow imports from the app package when running as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import get_settings


# ── Calibration knobs ─────────────────────────────────────────────────────────

MIN_PERSONS       = 5          # skip datasets with fewer unique persons
MIN_GENUINE_PAIRS = 10         # skip if not enough same-person pairs to fit
IMPOSTOR_SAMPLES  = 50_000     # cap for cross-person random pairs
TIER1_MARGIN      = 0.05       # tier1 = tier2 − margin (permissive pre-gate)
TIER2_FLOOR       = 0.18       # never go below this even if FAR allows it
TIER2_CEIL        = 0.60       # never go above this (avoids no-match deadlock)
TIER1_FLOOR       = 0.12       # absolute floor for tier1
DEFAULT_TARGET_FAR = 1e-4      # 1-in-10 000 impostor pair accepted

logger = logging.getLogger("calibrate_thresholds")


# ── Data loading ──────────────────────────────────────────────────────────────

async def _fetch_active_datasets(session) -> list[dict]:
    rows = (await session.execute(text("""
        SELECT dataset_id::text AS dataset_id,
               client_id::text  AS client_id,
               name,
               person_count
        FROM face_datasets
        WHERE status = 'ACTIVE'
    """))).fetchall()
    return [dict(r._mapping) for r in rows]


async def _fetch_embeddings(session, dataset_id: str) -> list[tuple[str, np.ndarray]]:
    """
    Return [(person_id, embedding)] for all ACTIVE embeddings in this dataset.
    pgvector returns the vector as text "[x1,x2,...]"; we parse it once here.
    """
    rows = (await session.execute(text("""
        SELECT person_id::text AS person_id, embedding::text AS embedding
        FROM face_embeddings
        WHERE dataset_id = (:did)::uuid
          AND is_active = true
    """), {"did": dataset_id})).fetchall()

    out: list[tuple[str, np.ndarray]] = []
    for r in rows:
        vec = np.fromstring(r.embedding.strip("[]"), sep=",", dtype=np.float32)
        n = float(np.linalg.norm(vec))
        if n > 1e-8:
            vec = vec / n
        out.append((r.person_id, vec))
    return out


# ── Pair sampling & threshold picking ─────────────────────────────────────────

def _build_distributions(
    pairs: list[tuple[str, np.ndarray]],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (genuine_sims, impostor_sims) as 1-D float arrays in [-1, 1].

    Genuine = every within-person pair (bounded by cap per person to avoid a
    few heavily-enrolled people dominating the distribution).
    Impostor = IMPOSTOR_SAMPLES random cross-person pairs.
    """
    by_person: dict[str, list[np.ndarray]] = {}
    for pid, emb in pairs:
        by_person.setdefault(pid, []).append(emb)

    # Genuine pairs — up to 50 per person, exhaustive combinations
    GENUINE_PER_PERSON_CAP = 50
    genuine: list[float] = []
    for pid, embs in by_person.items():
        if len(embs) < 2:
            continue
        idx = np.arange(len(embs))
        if len(idx) > 10:  # subsample huge galleries before enumerating
            idx = rng.choice(idx, size=10, replace=False)
        mat = np.stack([embs[i] for i in idx], axis=0)
        sim = mat @ mat.T
        # upper triangle, excluding the diagonal
        iu = np.triu_indices(sim.shape[0], k=1)
        vals = sim[iu].tolist()
        if len(vals) > GENUINE_PER_PERSON_CAP:
            vals = rng.choice(vals, size=GENUINE_PER_PERSON_CAP, replace=False).tolist()
        genuine.extend(float(v) for v in vals)

    # Impostor pairs — sample without allowing same-person pairs
    all_embs = np.stack([e for _, e in pairs], axis=0)
    all_pids = np.array([pid for pid, _ in pairs])
    n = len(all_embs)

    impostor: list[float] = []
    target = min(IMPOSTOR_SAMPLES, n * (n - 1) // 2)
    attempts = 0
    max_attempts = target * 3
    while len(impostor) < target and attempts < max_attempts:
        a = rng.integers(0, n, size=min(4096, target - len(impostor)))
        b = rng.integers(0, n, size=a.size)
        mask = (a != b) & (all_pids[a] != all_pids[b])
        a = a[mask]; b = b[mask]
        if a.size:
            sims = np.einsum("ij,ij->i", all_embs[a], all_embs[b])
            impostor.extend(float(v) for v in sims.tolist())
        attempts += a.size or 1

    return np.asarray(genuine, dtype=np.float64), np.asarray(impostor, dtype=np.float64)


def _pick_tier2(impostor: np.ndarray, genuine: np.ndarray, target_far: float) -> tuple[float, float]:
    """
    Tier-2 = the smallest τ such that (# impostor ≥ τ) / len(impostor) ≤ target_far.
    Returns (threshold, achieved_far).  Clamped to [TIER2_FLOOR, TIER2_CEIL].
    """
    if impostor.size == 0:
        return TIER2_FLOOR, 0.0

    # Quantile gives us the (1 - far)-percentile of the impostor distribution;
    # any threshold at or above that value accepts ≤ far fraction of impostors.
    q = 1.0 - target_far
    tau = float(np.quantile(impostor, q))
    tau = max(TIER2_FLOOR, min(TIER2_CEIL, tau))
    achieved_far = float(np.mean(impostor >= tau))
    return tau, achieved_far


def _frr_at(genuine: np.ndarray, tau: float) -> float:
    if genuine.size == 0:
        return 0.0
    return float(np.mean(genuine < tau))


def _percentiles(arr: np.ndarray) -> dict[str, float]:
    if arr.size == 0:
        return {"p01": 0.0, "p05": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    qs = np.quantile(arr, [0.01, 0.05, 0.50, 0.95, 0.99]).tolist()
    return {
        "p01": float(qs[0]), "p05": float(qs[1]), "p50": float(qs[2]),
        "p95": float(qs[3]), "p99": float(qs[4]),
    }


# ── Per-dataset calibration ───────────────────────────────────────────────────

async def _calibrate_one(
    session,
    ds: dict,
    target_far: float,
    dry_run: bool,
    rng: np.random.Generator,
) -> dict[str, Any]:
    dataset_id = ds["dataset_id"]
    embs = await _fetch_embeddings(session, dataset_id)
    n_embs = len(embs)
    n_persons = len({pid for pid, _ in embs})

    if n_persons < MIN_PERSONS:
        return {
            "dataset_id": dataset_id, "name": ds.get("name"),
            "skipped": f"only {n_persons} persons (< {MIN_PERSONS})",
        }

    genuine, impostor = _build_distributions(embs, rng)

    if genuine.size < MIN_GENUINE_PAIRS:
        return {
            "dataset_id": dataset_id, "name": ds.get("name"),
            "skipped": f"only {genuine.size} genuine pairs (< {MIN_GENUINE_PAIRS})",
        }

    tier2, achieved_far = _pick_tier2(impostor, genuine, target_far)
    tier1 = max(TIER1_FLOOR, tier2 - TIER1_MARGIN)
    frr = _frr_at(genuine, tier2)

    stats = {
        "target_far": target_far,
        "achieved_far": achieved_far,
        "frr_at_tier2": frr,
        "n_persons": n_persons,
        "n_embeddings": n_embs,
        "n_genuine_pairs": int(genuine.size),
        "n_impostor_pairs": int(impostor.size),
        "genuine_percentiles": _percentiles(genuine),
        "impostor_percentiles": _percentiles(impostor),
        "tier1_margin": TIER1_MARGIN,
        "calibrator_version": 1,
    }

    if not dry_run:
        await session.execute(text("""
            UPDATE face_datasets
            SET tier1_threshold   = :t1,
                tier2_threshold   = :t2,
                calibration_stats = CAST(:stats AS JSONB),
                calibrated_at     = :ts
            WHERE dataset_id = (:did)::uuid
        """), {
            "t1":    tier1,
            "t2":    tier2,
            "stats": json.dumps(stats),
            "ts":    int(time.time()),
            "did":   dataset_id,
        })

    return {
        "dataset_id": dataset_id,
        "name":       ds.get("name"),
        "tier1":      tier1,
        "tier2":      tier2,
        "stats":      stats,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

async def main_async(args: argparse.Namespace) -> int:
    settings = get_settings()
    engine = create_async_engine(settings.database_url, echo=False)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    rng = np.random.default_rng(seed=args.seed)
    results: list[dict] = []

    try:
        async with factory() as session:
            async with session.begin():
                if args.dataset_id:
                    targets = [{
                        "dataset_id": args.dataset_id,
                        "client_id":  None,
                        "name":       None,
                        "person_count": None,
                    }]
                else:
                    targets = await _fetch_active_datasets(session)

                print(f"[info] calibrating {len(targets)} dataset(s) "
                      f"(target_far={args.target_far}, dry_run={args.dry_run})")

                for ds in targets:
                    try:
                        res = await _calibrate_one(
                            session, ds, args.target_far, args.dry_run, rng,
                        )
                    except Exception as exc:  # never abort the whole batch
                        logger.exception("calibration failed for %s", ds["dataset_id"])
                        res = {"dataset_id": ds["dataset_id"], "error": str(exc)}
                    results.append(res)
                    _print_one(res)
    finally:
        await engine.dispose()

    if args.json:
        print(json.dumps(results, indent=2))

    return 0


def _print_one(res: dict) -> None:
    did = res.get("dataset_id", "?")
    name = res.get("name") or "(unnamed)"
    if "error" in res:
        print(f"  ✗ {name} [{did[:8]}] — error: {res['error']}")
        return
    if "skipped" in res:
        print(f"  ⊘ {name} [{did[:8]}] — skipped: {res['skipped']}")
        return
    s = res["stats"]
    print(
        f"  ✓ {name} [{did[:8]}] "
        f"tier1={res['tier1']:.3f} tier2={res['tier2']:.3f}  "
        f"FAR={s['achieved_far']:.2e} FRR={s['frr_at_tier2']:.3f}  "
        f"(persons={s['n_persons']} pairs g={s['n_genuine_pairs']} i={s['n_impostor_pairs']})"
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--dataset-id", help="Calibrate a single dataset (uuid)")
    g.add_argument("--all", action="store_true", help="Calibrate every ACTIVE dataset")
    p.add_argument("--target-far", type=float, default=DEFAULT_TARGET_FAR,
                   help=f"Target False-Accept Rate (default {DEFAULT_TARGET_FAR})")
    p.add_argument("--seed", type=int, default=12345, help="RNG seed for pair sampling")
    p.add_argument("--dry-run", action="store_true",
                   help="Compute & print thresholds without writing to the DB")
    p.add_argument("--json", action="store_true",
                   help="Also print full results as JSON at the end")
    args = p.parse_args()

    rc = asyncio.run(main_async(args))
    sys.exit(rc)


if __name__ == "__main__":
    main()
