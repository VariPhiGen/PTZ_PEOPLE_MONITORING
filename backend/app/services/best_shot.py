"""
Best-Shot Gallery for ACAS.

Maintains the top-K face crops per identity per session, ranked by quality score.
Used for: dashboard thumbnails, re-enrollment quality, audit trail.

Only updates when a new capture beats the current worst in the gallery.
Crops are stored in MinIO; metadata is kept in memory (not persisted to DB in
this implementation — a future Alembic migration adds the best_shots table).

Config (environment variables):
  BEST_SHOT_GALLERY_SIZE  5
  BEST_SHOT_MIN_QUALITY   0.6   (do not store crops below this quality)
"""
from __future__ import annotations

import io
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

BEST_SHOT_GALLERY_SIZE: int   = int(  os.getenv("BEST_SHOT_GALLERY_SIZE", "5"))
BEST_SHOT_MIN_QUALITY:  float = float(os.getenv("BEST_SHOT_MIN_QUALITY",  "0.6"))


@dataclass(order=False)
class BestShotEntry:
    quality_score:  float
    face_crop_url:  str           # MinIO object key
    captured_at:    float         # monotonic
    yaw_degrees:    float
    pitch_degrees:  float
    camera_id:      Optional[str] = None
    preset_id:      Optional[str] = None


@dataclass
class IdentityGallery:
    """Gallery of best-shot entries for one identity in one session."""
    identity_id: str
    session_id:  str
    entries:     list[BestShotEntry] = field(default_factory=list)

    @property
    def worst_score(self) -> float:
        if not self.entries:
            return -1.0
        return min(e.quality_score for e in self.entries)

    @property
    def best_entry(self) -> Optional[BestShotEntry]:
        if not self.entries:
            return None
        return max(self.entries, key=lambda e: e.quality_score)


class BestShotGallery:
    """
    Top-K face crop gallery per identity per session.

    Usage:
        gallery = BestShotGallery(minio_client, bucket="acas-bestshots")
        await gallery.maybe_update(
            identity_id, session_id, face_crop, quality_score,
            yaw=yaw, pitch=pitch, camera_id=cam, preset_id=preset
        )
        url = gallery.get_best_url(identity_id, session_id)
    """

    def __init__(
        self,
        minio_client=None,
        bucket: str   = "acas-bestshots",
        gallery_size:  int   = BEST_SHOT_GALLERY_SIZE,
        min_quality:   float = BEST_SHOT_MIN_QUALITY,
    ) -> None:
        self._minio       = minio_client
        self._bucket      = bucket
        self._gallery_size = gallery_size
        self._min_quality  = min_quality
        # (identity_id, session_id) → IdentityGallery
        self._galleries:  dict[tuple[str, str], IdentityGallery] = {}

    async def maybe_update(
        self,
        identity_id:  str,
        session_id:   str,
        face_crop:    np.ndarray,
        quality_score: float,
        *,
        yaw:       float = 0.0,
        pitch:     float = 0.0,
        camera_id: Optional[str] = None,
        preset_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Attempt to add this face crop to the gallery.

        Only stored if:
          - quality >= min_quality
          - gallery not full, OR this crop beats the current worst entry

        Returns the MinIO object URL if stored, None otherwise.
        """
        if quality_score < self._min_quality:
            return None

        key = (identity_id, session_id)
        gal = self._galleries.get(key)
        if gal is None:
            gal = IdentityGallery(identity_id=identity_id, session_id=session_id)
            self._galleries[key] = gal

        # Check if we should store this crop
        if len(gal.entries) >= self._gallery_size and quality_score <= gal.worst_score:
            return None

        # Upload to MinIO
        url = await self._upload(identity_id, session_id, face_crop, quality_score)
        if url is None:
            return None

        entry = BestShotEntry(
            quality_score=quality_score,
            face_crop_url=url,
            captured_at=time.monotonic(),
            yaw_degrees=yaw,
            pitch_degrees=pitch,
            camera_id=camera_id,
            preset_id=preset_id,
        )

        if len(gal.entries) < self._gallery_size:
            gal.entries.append(entry)
        else:
            # Replace worst entry
            worst_idx = min(range(len(gal.entries)), key=lambda i: gal.entries[i].quality_score)
            # Optionally delete old object from MinIO
            await self._delete(gal.entries[worst_idx].face_crop_url)
            gal.entries[worst_idx] = entry

        return url

    def get_best_url(
        self, identity_id: str, session_id: str
    ) -> Optional[str]:
        """Return the URL of the highest-quality face crop for this identity/session."""
        gal = self._galleries.get((identity_id, session_id))
        if gal is None or not gal.entries:
            return None
        best = gal.best_entry
        return best.face_crop_url if best else None

    def get_gallery(
        self, identity_id: str, session_id: str
    ) -> list[BestShotEntry]:
        gal = self._galleries.get((identity_id, session_id))
        return gal.entries if gal else []

    def on_session_end(self, session_id: str) -> None:
        """Remove all in-memory gallery entries for a session."""
        self._galleries = {
            k: v for k, v in self._galleries.items()
            if k[1] != session_id
        }

    # ── MinIO helpers ──────────────────────────────────────────────────────────

    async def _upload(
        self,
        identity_id:  str,
        session_id:   str,
        face_crop:    np.ndarray,
        quality:      float,
    ) -> Optional[str]:
        """Encode crop as JPEG and upload to MinIO.  Returns object key or None."""
        if self._minio is None:
            return None

        try:
            import cv2
            ok, buf = cv2.imencode(".jpg", face_crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if not ok:
                return None

            obj_key = (
                f"bestshots/{identity_id}/{session_id}/"
                f"{uuid.uuid4().hex[:8]}_q{int(quality*100):03d}.jpg"
            )

            import asyncio
            await asyncio.to_thread(
                self._minio.put_object,
                self._bucket,
                obj_key,
                io.BytesIO(buf.tobytes()),
                length=len(buf),
                content_type="image/jpeg",
            )
            return obj_key
        except Exception as exc:
            logger.debug("BestShot MinIO upload failed: %s", exc)
            return None

    async def _delete(self, obj_key: str) -> None:
        """Best-effort delete of a MinIO object."""
        if self._minio is None or not obj_key:
            return
        try:
            import asyncio
            await asyncio.to_thread(self._minio.remove_object, self._bucket, obj_key)
        except Exception:
            pass
