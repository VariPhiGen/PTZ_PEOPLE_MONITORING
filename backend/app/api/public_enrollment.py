"""
Public self-enrollment API — no JWT required.

Accessed via a client-specific enrollment token embedded in the URL.
Admins generate tokens via POST /api/enrollment-tokens and share the link:

    https://<host>/enroll/<token>

Routes:
    GET  /api/public/enroll/{token}          — validate token, return org info
    POST /api/public/enroll/{token}/upload   — quality-check one face image
    POST /api/public/enroll/{token}/submit   — create person + enroll all images
"""
from __future__ import annotations

import asyncio
import base64
import logging
import secrets
import time
import uuid

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, EmailStr
from sqlalchemy import text

from app.deps import DBSession

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/public/enroll", tags=["public-enrollment"])

# ── Rate limiting constants ────────────────────────────────────────────────────
_UPLOAD_RL_KEY = "rl:public_enroll:upload:{token}"
_SUBMIT_RL_KEY = "rl:public_enroll:submit:{ip}"
_UPLOAD_MAX    = 30   # uploads per minute per token
_SUBMIT_MAX    = 10   # submits per hour per IP


# ── Schemas ───────────────────────────────────────────────────────────────────

class PublicSubmitRequest(BaseModel):
    name:        str
    external_id: str | None = None
    email:       str | None = None
    phone:       str | None = None
    department:  str | None = None
    image_keys:  list[str]          # temp_keys from /upload


# ── Internal helpers ──────────────────────────────────────────────────────────

async def _validate_token(token: str, db: DBSession) -> dict:
    """
    Validate enrollment token and return its row as a dict.
    Raises 404 / 410 on invalid / expired / exhausted tokens.
    """
    row = (
        await db.execute(
            text("""
                SELECT et.token_id, et.client_id, et.token, et.label,
                       et.role_default, et.dataset_id,
                       et.expires_at, et.max_uses, et.use_count,
                       c.name AS client_name
                FROM enrollment_tokens et
                JOIN clients c ON c.client_id = et.client_id
                WHERE et.token = :t AND et.is_active = TRUE
            """),
            {"t": token},
        )
    ).mappings().one_or_none()

    if row is None:
        raise HTTPException(status_code=404, detail="Enrollment link not found or inactive")

    row = dict(row)
    now = int(time.time())
    if row["expires_at"] and now > row["expires_at"]:
        raise HTTPException(status_code=410, detail="Enrollment link has expired")
    if row["max_uses"] and row["use_count"] >= row["max_uses"]:
        raise HTTPException(status_code=410, detail="Enrollment link usage limit reached")

    return row


async def _rate_limit(redis, key: str, max_calls: int, window_s: int) -> None:
    """Sliding-window rate limit. Silently skips if Redis is unavailable."""
    if redis is None:
        return
    try:
        pipe = redis.pipeline()
        pipe.incr(key)
        pipe.expire(key, window_s)
        results = await asyncio.to_thread(pipe.execute)
        if results[0] > max_calls:
            raise HTTPException(status_code=429, detail="Too many requests — try again shortly")
    except HTTPException:
        raise
    except Exception:
        pass


def _quality_reason_human(reason: str) -> str:
    """Convert API reason codes to user-friendly messages."""
    mapping = {
        "no_face_detected":  "No face detected — make sure your face is clearly visible",
        "multiple_faces":    "Multiple faces in frame — only one person per photo",
        "low_confidence":    "Face not clear enough — improve lighting or move closer",
        "face_too_small":    "Face too small — move closer to the camera",
        "blurry":            "Image is blurry — hold still and try again",
        "file_too_large":    "File too large — max 10 MB",
        "unsupported_format": "Unsupported format — use JPEG or PNG",
        "cannot_decode":     "Cannot read image — try again",
    }
    for k, v in mapping.items():
        if reason.startswith(k):
            return v
    return "Photo quality too low — please try again"


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/{token}")
async def get_enrollment_info(token: str, db: DBSession) -> dict:
    """
    Validate an enrollment token and return organisation info for the landing page.
    Called immediately when the user opens the enrollment URL.
    """
    tok = await _validate_token(token, db)
    return {
        "valid":        True,
        "client_name":  tok["client_name"],
        "label":        tok["label"] or "Face Enrollment",
        "role_default": tok["role_default"],
        "expires_at":   tok["expires_at"],
        "uses_remaining": (
            tok["max_uses"] - tok["use_count"]
            if tok["max_uses"] else None
        ),
    }


@router.post("/{token}/upload")
async def public_upload(
    token:   str,
    db:      DBSession,
    request: Request,
    file:    UploadFile = File(...),
) -> dict:
    """
    Quality-check a single face image captured from the webcam.
    Returns ``temp_key`` on success — pass it back in the /submit call.

    The image is NOT stored permanently here; it is returned as a base64 blob
    that the frontend holds until /submit is called.
    """
    tok      = await _validate_token(token, db)
    redis    = getattr(request.app.state, "redis", None)
    pipeline = getattr(request.app.state, "ai_pipeline", None)

    await _rate_limit(
        redis,
        _UPLOAD_RL_KEY.format(token=token),
        _UPLOAD_MAX,
        60,
    )

    image_bytes = await file.read()
    filename    = file.filename or "capture.jpg"

    # Size check
    if len(image_bytes) > 10 * 1024 * 1024:
        return {"passed": False, "reason": "file_too_large",
                "reason_human": "File too large — max 10 MB", "temp_key": None}

    # Format check
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ("jpg", "jpeg", "png", "webp"):
        # Accept webp from webcam captures
        ct = file.content_type or ""
        if "jpeg" not in ct and "png" not in ct and "webp" not in ct:
            return {"passed": False, "reason": "unsupported_format",
                    "reason_human": "Unsupported format", "temp_key": None}

    buf   = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if frame is None:
        return {"passed": False, "reason": "cannot_decode",
                "reason_human": "Cannot read image — try again", "temp_key": None}

    quality_info: dict = {}

    if pipeline is not None:
        try:
            faces = await asyncio.to_thread(pipeline.detect_faces, frame, None)

            if len(faces) == 0:
                return {"passed": False, "reason": "no_face_detected",
                        "reason_human": _quality_reason_human("no_face_detected"),
                        "temp_key": None, "quality": {}}

            if len(faces) > 1:
                return {"passed": False, "reason": f"multiple_faces:{len(faces)}",
                        "reason_human": _quality_reason_human("multiple_faces"),
                        "temp_key": None, "quality": {}}

            face = faces[0]

            if face.conf < 0.50:
                return {"passed": False, "reason": f"low_confidence:{face.conf:.2f}",
                        "reason_human": _quality_reason_human("low_confidence"),
                        "temp_key": None, "quality": {}}

            if face.inter_ocular_px < 20:
                return {"passed": False, "reason": f"face_too_small:{face.inter_ocular_px:.1f}px",
                        "reason_human": _quality_reason_human("face_too_small"),
                        "temp_key": None, "quality": {}}

            try:
                chip = await asyncio.to_thread(pipeline.align_face, frame, face.landmarks)
                sharpness = float(
                    cv2.Laplacian(cv2.cvtColor(chip, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
                )
                if sharpness < 10:
                    return {"passed": False, "reason": f"blurry:{sharpness:.1f}",
                            "reason_human": _quality_reason_human("blurry"),
                            "temp_key": None, "quality": {"sharpness": round(sharpness, 1)}}
                quality_info = {
                    "iod_px":     round(face.inter_ocular_px, 1),
                    "sharpness":  round(sharpness, 1),
                    "confidence": round(face.conf, 3),
                }
            except Exception:
                quality_info = {"iod_px": round(face.inter_ocular_px, 1)}

        except Exception as exc:
            logger.warning("Public enroll quality check error: %s", exc)
            quality_info = {"note": "quality_check_unavailable"}
    else:
        quality_info = {"note": "pipeline_not_loaded"}

    b64      = base64.b64encode(image_bytes).decode()
    temp_key = f"b64:{b64}"

    return {
        "passed":       True,
        "reason":       "OK",
        "reason_human": "Great shot!",
        "temp_key":     temp_key,
        "quality":      quality_info,
    }


@router.post("/{token}/submit", status_code=201)
async def public_submit(
    token:   str,
    db:      DBSession,
    request: Request,
    body:    PublicSubmitRequest,
) -> dict:
    """
    Create person record and enroll all face images in one shot.
    Increments the token's use_count after a successful enrollment.
    """
    from app.models.persons import Person, PersonRole, PersonStatus

    tok      = await _validate_token(token, db)
    redis    = getattr(request.app.state, "redis", None)
    face_repo = getattr(request.app.state, "face_repo", None)

    if face_repo is None:
        raise HTTPException(status_code=503, detail="Enrollment service unavailable — try again shortly")

    if not body.image_keys:
        raise HTTPException(status_code=422, detail="No images provided")

    if len(body.image_keys) > 20:
        raise HTTPException(status_code=422, detail="Maximum 20 images per enrollment")

    # IP-based rate limit on submit
    client_ip = request.client.host if request.client else "unknown"
    await _rate_limit(redis, _SUBMIT_RL_KEY.format(ip=client_ip), _SUBMIT_MAX, 3600)

    client_id  = str(tok["client_id"])
    dataset_id = tok["dataset_id"]

    # Resolve default dataset if token has none
    if not dataset_id:
        ds = (
            await db.execute(
                text("""
                    SELECT dataset_id::text FROM face_datasets
                    WHERE client_id = (:cid)::uuid AND is_default = TRUE
                    LIMIT 1
                """),
                {"cid": client_id},
            )
        ).fetchone()
        if ds:
            dataset_id = ds[0]

    # ── Decode images → raw JPEG bytes (enroll_person expects list[bytes]) ───────
    images: list[bytes] = []

    for key in body.image_keys:
        if not key.startswith("b64:"):
            logger.warning("Public enroll: unexpected temp_key format: %s", key[:20])
            continue
        raw = base64.b64decode(key[4:])
        # Validate the image can be decoded, then keep the original bytes
        buf = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is not None:
            images.append(raw)

    if not images:
        raise HTTPException(status_code=422, detail="No valid images could be decoded")

    # ── Create person ─────────────────────────────────────────────────────────
    now = int(time.time())

    # Deduplicate external_id — if supplied, check it doesn't already exist
    ext_id = body.external_id
    if not ext_id:
        ext_id = f"self-{secrets.token_hex(6)}"

    existing = (
        await db.execute(
            text("""
                SELECT person_id::text FROM persons
                WHERE client_id = (:cid)::uuid AND external_id = :eid
                LIMIT 1
            """),
            {"cid": client_id, "eid": ext_id},
        )
    ).fetchone()

    if existing:
        raise HTTPException(
            status_code=409,
            detail="A person with this ID already exists in this organisation",
        )

    try:
        role = PersonRole(tok["role_default"])
    except ValueError:
        role = PersonRole.STUDENT

    person = Person(
        person_id   = uuid.uuid4(),
        client_id   = uuid.UUID(client_id),
        dataset_id  = uuid.UUID(dataset_id) if dataset_id else None,
        name        = body.name.strip(),
        external_id = ext_id,
        email       = body.email,
        phone       = body.phone,
        department  = body.department,
        role        = role,
        status      = PersonStatus.ACTIVE,
        consent_at  = now,
        created_at  = now,
        updated_at  = now,
    )
    db.add(person)
    # Commit person first — face_repo.enroll_person uses its own DB connection
    # and needs the person row to be visible (FK on face_embeddings.person_id).
    await db.commit()

    # ── Run embedding + FAISS index ───────────────────────────────────────────
    try:
        result = await face_repo.enroll_person(
            client_id  = client_id,
            person_id  = str(person.person_id),
            images     = images,
            metadata   = {},
            dataset_id = dataset_id,
        )
    except Exception as exc:
        # Person was committed — clean it up before surfacing the error
        await db.execute(
            text("DELETE FROM persons WHERE person_id = (:pid)::uuid"),
            {"pid": str(person.person_id)},
        )
        await db.commit()
        logger.error("Public enroll embedding failed: %s", exc)
        raise HTTPException(status_code=500, detail="Face embedding failed — please retake photos and try again")

    if result.error:
        await db.execute(
            text("DELETE FROM persons WHERE person_id = (:pid)::uuid"),
            {"pid": str(person.person_id)},
        )
        await db.commit()
        raise HTTPException(status_code=422, detail=f"Enrollment failed: {result.error}")

    # ── Increment token use_count ─────────────────────────────────────────────
    await db.execute(
        text("UPDATE enrollment_tokens SET use_count = use_count + 1 WHERE token = :t"),
        {"t": token},
    )
    await db.commit()

    logger.info(
        "Public self-enrollment: person=%s client=%s images=%d",
        person.person_id, client_id, len(images),
    )

    return {
        "person_id":      str(person.person_id),
        "name":           person.name,
        "images_enrolled": result.images_passed,
        "message":        "Enrollment successful!",
    }
