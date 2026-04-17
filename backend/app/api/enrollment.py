"""
Enrollment API — face enrollment, quality-gated image upload, and management.

POST /api/enrollment/upload       — validate & temporarily store image(s) for a person
POST /api/enrollment/enroll       — run full enrollment (quality gate + FAISS rebuild)
POST /api/enrollment/bulk-import  — CSV manifest + image ZIP, async
GET  /api/enrollment/guidelines   — quality requirements (no auth required)
GET  /api/enrollment/list         — paginated enrolled persons
GET  /api/enrollment/{person_id}  — enrollment detail
GET  /api/enrollment/{person_id}/images — enrollment image URLs
PUT  /api/enrollment/{person_id}/re-enroll — replace embeddings
DELETE /api/enrollment/{person_id} — remove all face data

All mutation routes require face_embeddings:create / face_embeddings:delete.
Read routes require face_embeddings:read.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, Body, File, Form, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel
from sqlalchemy import func, select, text

from app.api._shared import audit, now_epoch, rate_limit, resolve_client_id
from app.deps import CurrentUser, DBSession
from app.middleware.auth import require_permission
from app.models.face_embeddings import FaceEmbedding
from app.models.persons import Person, PersonRole, PersonStatus

router = APIRouter(prefix="/api/enrollment", tags=["enrollment"])

_RD  = require_permission("face_embeddings:read")
_WR  = require_permission("face_embeddings:create")
_DEL = require_permission("face_embeddings:delete")

_UPLOAD_RL_KEY    = "rl:enrollment:upload:{client_id}"
_UPLOAD_MAX       = 60
_UPLOAD_WINDOW_S  = 60   # 60 uploads/min per client


# ── Schemas ───────────────────────────────────────────────────────────────────

class EnrollRequest(BaseModel):
    # If person_id is omitted the endpoint will create the person from the inline fields
    person_id:   str | None = None
    client_id:   str | None = None
    dataset_id:  str | None = None   # target face dataset (uses default if omitted)
    image_keys:  list[str]
    # Inline person creation fields (used when person_id is None)
    name:        str | None = None
    external_id: str | None = None
    role:        str | None = None
    department:  str | None = None
    email:       str | None = None
    phone:       str | None = None


class BulkImportRequest(BaseModel):
    client_id:   str | None = None
    manifest:    list[dict]           # [{person_id, name, role, images: [key, ...]}]


class PersonCreateInline(BaseModel):
    """Used when enrolling a new person inline (without pre-creating via /api/persons)."""
    name:       str
    role:       str       = PersonRole.STUDENT
    department: str | None = None
    external_id: str | None = None
    email:      str | None = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _embedding_summary(emb: FaceEmbedding) -> dict:
    return {
        "embedding_id":   str(emb.embedding_id),
        "person_id":      str(emb.person_id),
        "client_id":      str(emb.client_id),
        "version":        emb.version,
        "source":         emb.source,
        "confidence_avg": emb.confidence_avg,
        "quality_score":  emb.quality_score,
        "is_active":      emb.is_active,
        "image_count":    len(emb.image_refs) if emb.image_refs else 0,
        "created_at":     emb.created_at,
    }


def _person_dict(
    p: Person,
    embedding: FaceEmbedding | None = None,
    *,
    total_image_count: int | None = None,
) -> dict:
    emb_summary = _embedding_summary(embedding) if embedding else None
    image_count = (
        total_image_count
        if total_image_count is not None
        else (emb_summary.get("image_count", 0) if emb_summary else 0)
    )
    d: dict[str, Any] = {
        "person_id":     str(p.person_id),
        "client_id":     str(p.client_id),
        "dataset_id":    p.dataset_id,
        "external_id":   p.external_id,
        "name":          p.name,
        "role":          p.role.value if hasattr(p.role, "value") else p.role,
        "department":    p.department,
        "email":         p.email,
        "phone":         p.phone,
        "status":        p.status.value if hasattr(p.status, "value") else p.status,
        "consent_at":    p.consent_at,
        "created_at":    p.created_at,
        "updated_at":    p.updated_at,
        "embedding":     emb_summary,
        "enrolled":      emb_summary is not None,
        "quality_score": emb_summary["quality_score"] if emb_summary else None,
        "image_count":   image_count,
        "thumbnail":     None,
        "last_seen":     None,
    }
    return d


async def _get_person_or_404(db: DBSession, person_id: str) -> Person:
    p = await db.get(Person, uuid.UUID(person_id))
    if p is None:
        raise HTTPException(status_code=404, detail="Person not found")
    return p


async def _active_embedding(db: DBSession, person_id: str) -> FaceEmbedding | None:
    result = await db.execute(
        select(FaceEmbedding)
        .where(
            FaceEmbedding.person_id == uuid.UUID(person_id),
            FaceEmbedding.is_active.is_(True),
        )
        .order_by(FaceEmbedding.created_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def _total_image_count(db: DBSession, person_id: str) -> int:
    """Sum image_refs lengths across ALL active embedding rows for a person."""
    row = await db.execute(
        text(
            "SELECT COALESCE(SUM(array_length(image_refs, 1)), 0) "
            "FROM face_embeddings "
            "WHERE person_id = :pid AND is_active = true"
        ),
        {"pid": person_id},
    )
    return int(row.scalar() or 0)


def _get_face_repo(request: Request) -> Any:
    repo = getattr(request.app.state, "face_repo", None)
    if repo is None:
        raise HTTPException(status_code=503, detail="Enrollment service unavailable — face_repo not initialised")
    return repo


def _get_pipeline(request: Request, *, required: bool = True) -> Any:
    pipeline = getattr(request.app.state, "ai_pipeline", None)
    if pipeline is None and required:
        raise HTTPException(status_code=503, detail="AI pipeline not initialised")
    return pipeline


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/guidelines")
async def enrollment_guidelines() -> dict:
    """Return quality requirements for enrollment images (public)."""
    return {
        "min_face_count":        1,
        "max_face_count":        1,
        "min_inter_ocular_px":   20,
        "min_sharpness_score":   10,
        "min_face_confidence":   0.50,
        "max_yaw_degrees":       45,
        "recommended_images":    5,
        "max_images_per_person": 20,
        "allowed_formats":       ["image/jpeg", "image/png"],
        "max_file_size_mb":      10,
        "tips": [
            "Front-facing, eyes open",
            "Even lighting — avoid harsh shadows",
            "No sunglasses or face coverings",
            "Resolution >= 640×480",
        ],
    }


@router.post("/upload", dependencies=[_WR])
async def upload_image(
    request:   Request,
    db:        DBSession,
    # person_id is optional — not yet created when doing pre-enrollment quality check
    person_id: str | None         = Form(None),
    # Accept either "file" (single) or "images" (multiple) field name
    file:      UploadFile | None  = File(None),
    images:    list[UploadFile]   = File(default=[]),
    client_id: str | None         = Form(None),
) -> dict | list:
    """
    Upload one or more face images and run the quality gate.
    Accepts either a single ``file`` field or a ``images`` list field.
    Returns a single result dict (single file) or a list of result dicts (multiple).
    ``person_id`` is optional — omit it during pre-enrollment quality checks.
    """
    import asyncio
    import io
    import numpy as np
    import cv2

    cid      = resolve_client_id(request, client_id, require=False) or "temp"
    pid      = person_id or "temp"
    pipeline = _get_pipeline(request, required=False)   # None = models not downloaded yet
    minio    = request.app.state.minio

    try:
        await rate_limit(
            request.app.state.redis,
            _UPLOAD_RL_KEY.format(client_id=cid),
            _UPLOAD_MAX,
            _UPLOAD_WINDOW_S,
        )
    except Exception:
        pass  # Don't block upload on Redis failure

    # Normalise to a list of (filename, bytes)
    uploads: list[tuple[str, bytes]] = []
    if file is not None:
        uploads.append((file.filename or "image.jpg", await file.read()))
    for img in images:
        uploads.append((img.filename or "image.jpg", await img.read()))

    if not uploads:
        raise HTTPException(status_code=422, detail="No image file(s) provided")

    async def _check_one(filename: str, image_bytes: bytes) -> dict:
        """Quality-gate a single image and store it in MinIO on pass."""
        # Validate size & type
        if len(image_bytes) > 10 * 1024 * 1024:
            return {"passed": False, "reason": "file_too_large", "temp_key": None, "filename": filename}

        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext not in ("jpg", "jpeg", "png"):
            return {"passed": False, "reason": "unsupported_format", "temp_key": None, "filename": filename}

        buf   = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            return {"passed": False, "reason": "cannot_decode", "temp_key": None, "filename": filename}

        quality_info: dict = {}

        # AI quality gate — skip gracefully when pipeline not loaded (models not yet downloaded)
        if pipeline is not None:
            try:
                faces = await asyncio.to_thread(pipeline.detect_faces, frame, None)
                if len(faces) == 0:
                    return {"passed": False, "reason": "no_face_detected", "temp_key": None, "filename": filename}
                if len(faces) > 1:
                    return {"passed": False, "reason": f"multiple_faces:{len(faces)}", "temp_key": None, "filename": filename}

                face = faces[0]
                # Confidence already filtered by SCRFD at 0.50 — warn but don't block
                if face.conf < 0.50:
                    return {"passed": False, "reason": f"low_confidence:{face.conf:.2f}", "temp_key": None, "filename": filename}

                # IOD < 20px means the face is genuinely too small to produce a useful embedding
                if face.inter_ocular_px < 20:
                    return {"passed": False, "reason": f"face_too_small:{face.inter_ocular_px:.1f}px_iod", "temp_key": None, "filename": filename}

                # Landmark geometry check — catches profile/occluded/upside-down faces
                from app.services.face_repository import _landmark_ok
                if not _landmark_ok(face.landmarks, face.inter_ocular_px):
                    return {"passed": False, "reason": "occlusion_or_profile", "temp_key": None, "filename": filename}

                # Yaw check — reject side-profile photos (> 45°)
                yaw_deg = pipeline.estimate_yaw(face.landmarks)
                if abs(yaw_deg) > 45.0:
                    return {"passed": False, "reason": f"yaw_too_large:{yaw_deg:.0f}deg", "temp_key": None, "filename": filename}

                try:
                    chip = await asyncio.to_thread(pipeline.align_face, frame, face.landmarks)
                    sharpness = float(cv2.Laplacian(
                        cv2.cvtColor(chip, cv2.COLOR_BGR2GRAY), cv2.CV_64F
                    ).var())
                    if sharpness < 10:
                        return {"passed": False, "reason": f"blurry:{sharpness:.1f}", "temp_key": None, "filename": filename}
                    quality_info = {
                        "iod_px":     round(face.inter_ocular_px, 1),
                        "sharpness":  round(sharpness, 1),
                        "confidence": round(face.conf, 3),
                        "yaw_deg":    round(yaw_deg, 1),
                    }
                except Exception:
                    quality_info = {"iod_px": round(face.inter_ocular_px, 1), "confidence": round(face.conf, 3)}
            except Exception as exc:
                # Pipeline error — pass with a warning so enrollment isn't blocked
                quality_info = {"warning": f"quality_check_unavailable:{exc}"}
        else:
            # No AI pipeline loaded yet (models not downloaded) — accept all valid images
            # Real quality gating will happen when the pipeline is available.
            quality_info = {"note": "pipeline_not_loaded_quality_check_skipped"}

        # Return the image as base64 so the enroll step can send it back directly.
        # This avoids MinIO storage entirely at the upload/quality-check stage —
        # the actual permanent storage happens only when the person is enrolled.
        import base64
        b64 = base64.b64encode(image_bytes).decode()
        temp_key = f"b64:{b64}"

        return {"passed": True, "reason": "OK", "temp_key": temp_key, "filename": filename, "quality": quality_info}

    results = await asyncio.gather(*[_check_one(fn, fb) for fn, fb in uploads])

    # Return single dict for single-file uploads (backward compat), list otherwise
    if len(results) == 1 and file is not None and not images:
        return results[0]
    return list(results)


@router.post("/enroll", status_code=201, dependencies=[_WR])
async def enroll_person(body: EnrollRequest, request: Request, db: DBSession) -> dict:
    """
    Run full enrollment for a person using pre-validated image keys from /upload.
    If person_id is omitted, creates the person record from inline fields first.
    Builds AdaFace embeddings and (re)builds the FAISS index for the client.
    """
    from app.models.persons import Person, PersonStatus, PersonRole
    cid       = resolve_client_id(request, body.client_id, require=False)
    face_repo = _get_face_repo(request)
    minio     = request.app.state.minio

    # Resolve effective dataset_id — fall back to client's default dataset.
    # Also use the dataset to derive client_id for SUPER_ADMIN with no body client_id.
    effective_dataset_id = body.dataset_id
    if not cid and effective_dataset_id:
        # SUPER_ADMIN with explicit dataset_id — derive client from the dataset
        from sqlalchemy import text as _t
        ds_row = (await db.execute(
            _t("SELECT client_id::text FROM face_datasets WHERE dataset_id = (:did)::uuid LIMIT 1"),
            {"did": effective_dataset_id},
        )).fetchone()
        if ds_row:
            cid = ds_row.client_id
    elif cid and effective_dataset_id:
        # Validate that the supplied dataset_id belongs to the resolved client
        from sqlalchemy import text as _t
        ds_row = (await db.execute(
            _t("SELECT client_id::text FROM face_datasets WHERE dataset_id = (:did)::uuid LIMIT 1"),
            {"did": effective_dataset_id},
        )).fetchone()
        if not ds_row:
            raise HTTPException(status_code=404, detail="Dataset not found")
        if ds_row.client_id != cid:
            raise HTTPException(
                status_code=403,
                detail="Dataset does not belong to this client",
            )

    if not effective_dataset_id and cid:
        from sqlalchemy import text as _t
        ds_row = (await db.execute(
            _t("SELECT dataset_id::text FROM face_datasets WHERE client_id = (:cid)::uuid AND is_default = true LIMIT 1"),
            {"cid": cid},
        )).fetchone()
        if ds_row:
            effective_dataset_id = ds_row.dataset_id

    # Auto-create person if person_id not supplied
    effective_pid = body.person_id
    if not effective_pid:
        if not body.name:
            raise HTTPException(status_code=422, detail="Either person_id or name is required")
        if not cid:
            raise HTTPException(
                status_code=422,
                detail="client_id required — include it in the request body or assign a dataset_id",
            )
        # Validate role enum before creating the person
        try:
            person_role = PersonRole(body.role) if body.role else PersonRole.STUDENT
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid role {body.role!r}. Valid values: {[r.value for r in PersonRole]}",
            )
        new_pid = uuid.uuid4()
        new_person = Person(
            person_id   = new_pid,
            client_id   = uuid.UUID(cid),
            dataset_id  = effective_dataset_id,
            name        = body.name,
            external_id = body.external_id or str(uuid.uuid4())[:8],
            role        = person_role,
            department  = body.department,
            email       = body.email,
            phone       = body.phone,
            status      = PersonStatus.ACTIVE,
        )
        db.add(new_person)
        # flush to send the INSERT within the current transaction; commit so that
        # face_repo (separate DB session) can see the person via its FK checks.
        # We do NOT call db.refresh() because person_id was generated in Python
        # and SQLAlchemy's session.begin() context manager disallows further
        # queries after a manual commit().
        await db.flush()
        await db.commit()
        effective_pid = str(new_pid)
    else:
        # Existing person_id path — always verify ownership
        person_row = await db.get(Person, uuid.UUID(effective_pid))
        if person_row is None:
            raise HTTPException(status_code=404, detail="Person not found")
        person_cid = str(person_row.client_id)
        if cid and person_cid != cid:
            raise HTTPException(
                status_code=403,
                detail="Person does not belong to this client",
            )
        # Derive cid if SUPER_ADMIN did not supply one
        if not cid:
            cid = person_cid

    # Resolve image bytes from either base64 ("b64:<data>") or MinIO key
    import asyncio, io, base64

    async def _get_img(key: str) -> bytes | None:
        if key.startswith("b64:"):
            try:
                return base64.b64decode(key[4:])
            except Exception:
                return None
        try:
            evidence_bucket = getattr(request.app.state.settings, "minio_evidence_bucket", "face-evidence")
            return await asyncio.to_thread(
                lambda: minio.get_object(evidence_bucket, key).read()
            )
        except Exception:
            return None

    images = [img for img in await asyncio.gather(*[_get_img(k) for k in body.image_keys]) if img]
    if not images:
        raise HTTPException(status_code=400, detail="No valid images found for provided keys")

    result = await face_repo.enroll_person(
        client_id  = cid,
        person_id  = effective_pid,
        images     = images,
        metadata   = {},
        dataset_id = effective_dataset_id,
    )
    if result.error:
        raise HTTPException(status_code=422, detail=result.error)

    audit(request, "ENROLL_PERSON", "person", effective_pid,
          {"images_passed": result.images_passed, "images_total": result.images_total})

    return {
        "person_id":       result.person_id,
        "embedding_id":    result.embedding_id,
        "images_total":    result.images_total,
        "images_passed":   result.images_passed,
        "centroid_quality": result.centroid_quality,
        "image_refs":      result.image_refs,
    }


@router.post("/bulk-import", status_code=202, dependencies=[_WR])
async def bulk_import(body: BulkImportRequest, request: Request, db: DBSession) -> dict:
    """
    Accept a manifest of person→image-key mappings and enqueue enrollment jobs.
    Returns immediately; enrollment runs asynchronously via Kafka.
    """
    cid = resolve_client_id(request, body.client_id)
    import json as _json
    try:
        producer = request.app.state.kafka_producer
        for entry in body.manifest:
            producer.produce(
                "enrollment.events",
                key=cid.encode(),
                value=_json.dumps({
                    "event_type": "BULK_ENROLL",
                    "client_id":  cid,
                    "person_id":  entry.get("person_id"),
                    "image_keys": entry.get("images", []),
                }).encode(),
            )
        producer.poll(0)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to enqueue: {exc}")

    audit(request, "BULK_IMPORT", "client", cid, {"count": len(body.manifest)})
    return {"queued": len(body.manifest), "status": "accepted"}


@router.get("/list", dependencies=[_RD])
async def list_enrolled(
    request:    Request,
    db:         DBSession,
    q:          str | None  = Query(None, description="Name search"),
    role:       str | None  = Query(None),
    department: str | None  = Query(None),
    dataset_id: str | None  = Query(None, description="Filter by face dataset"),
    client_id:  str | None  = Query(None, description="Client scope (SUPER_ADMIN)"),
    limit:      int         = Query(50, ge=1, le=200),
    offset:     int         = Query(0, ge=0),
) -> dict:
    """List all persons with active face embeddings for the current client."""
    cid = resolve_client_id(request, client_id, require=False)
    query = (
        select(Person)
        .where(Person.status == PersonStatus.ACTIVE)
        .join(
            FaceEmbedding,
            (FaceEmbedding.person_id == Person.person_id)
            & FaceEmbedding.is_active.is_(True),
            isouter=False,
        )
    )
    if cid:
        query = query.where(Person.client_id == uuid.UUID(cid))
    if dataset_id:
        query = query.where(Person.dataset_id == uuid.UUID(dataset_id))
    if q:
        query = query.where(
            text("similarity(persons.name, :q) > 0.1").bindparams(q=q)
        )
    if role:
        query = query.where(Person.role == role)
    if department:
        query = query.where(Person.department.ilike(f"%{department}%"))

    person_count = await db.scalar(select(func.count()).select_from(query.distinct().subquery())) or 0
    result = await db.execute(
        query.distinct().order_by(Person.created_at.desc()).offset(offset).limit(limit)
    )
    persons = result.scalars().all()

    items = []
    for p in persons:
        pid = str(p.person_id)
        emb = await _active_embedding(db, pid)
        img_count = await _total_image_count(db, pid)
        items.append(_person_dict(p, emb, total_image_count=img_count))

    return {
        "total":  person_count,
        "offset": offset,
        "limit":  limit,
        "items":  items,
    }


@router.get("/image-file/{client_id}/{person_id}/{filename}")
async def serve_local_image(client_id: str, person_id: str, filename: str, request: Request):
    """Serve a locally stored enrollment image when MinIO was unavailable."""
    import pathlib
    from fastapi.responses import FileResponse
    base_dir = getattr(request.app.state.settings, "enrollment_fallback_dir", "/enrollment-images")
    fpath = pathlib.Path(base_dir) / client_id / person_id / filename
    if not fpath.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(fpath, media_type="image/jpeg")


@router.get("/thumbnail/{person_id}")
async def serve_person_thumbnail(person_id: str, request: Request):
    """
    Proxy the first active enrollment photo for a person through the backend,
    so the browser never needs a direct connection to MinIO.
    """
    import asyncio
    from fastapi.responses import Response

    db_factory = request.app.state.session_factory
    async with db_factory() as db:
        row = (await db.execute(
            text("""
                SELECT fe.image_refs[1] AS ref
                FROM face_embeddings fe
                WHERE fe.person_id = (:pid)::uuid
                  AND fe.is_active = true
                  AND array_length(fe.image_refs, 1) > 0
                ORDER BY fe.created_at ASC LIMIT 1
            """),
            {"pid": person_id},
        )).fetchone()

    if not row or not row.ref:
        raise HTTPException(status_code=404, detail="No thumbnail available")

    ref = row.ref

    # Local fallback image
    if ref.startswith("local:"):
        import pathlib
        from fastapi.responses import FileResponse
        raw_key = ref[6:]
        base_dir = getattr(request.app.state.settings, "enrollment_fallback_dir", "/enrollment-images")
        fpath = pathlib.Path(base_dir) / raw_key
        if not fpath.is_file():
            raise HTTPException(status_code=404, detail="Image not found")
        return FileResponse(fpath, media_type="image/jpeg")

    # MinIO image — proxy bytes through the backend
    minio = request.app.state.minio
    try:
        data = await asyncio.to_thread(
            lambda: minio.get_object("face-enrollment", ref).read()
        )
        return Response(content=data, media_type="image/jpeg",
                        headers={"Cache-Control": "max-age=3600"})
    except Exception:
        raise HTTPException(status_code=404, detail="Image not found")


@router.get("/{person_id}", dependencies=[_RD])
async def get_enrollment(person_id: str, db: DBSession) -> dict:
    p     = await _get_person_or_404(db, person_id)
    emb   = await _active_embedding(db, person_id)
    total = await _total_image_count(db, person_id)
    return _person_dict(p, emb, total_image_count=total)


@router.put("/{person_id}", dependencies=[_WR])
async def update_enrollment(
    person_id: str,
    request:   Request,
    db:        DBSession,
    body:      dict = Body(...),
) -> dict:
    """Update a person's profile fields (name, department, email, phone, status)."""
    p = await _get_person_or_404(db, person_id)
    allowed = {"name", "department", "email", "phone", "status", "role", "external_id"}
    updates = {k: v for k, v in body.items() if k in allowed and v is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="No valid fields to update")
    for k, v in updates.items():
        setattr(p, k, v)
    p.updated_at = int(time.time())
    await db.flush()
    emb   = await _active_embedding(db, person_id)
    total = await _total_image_count(db, person_id)
    await db.commit()
    audit(request, "UPDATE_PERSON", "person", person_id, updates)
    return _person_dict(p, emb, total_image_count=total)


@router.get("/{person_id}/images", dependencies=[_RD])
async def get_enrollment_images(person_id: str, request: Request, db: DBSession) -> dict:
    """Return image URLs for all enrollment images (MinIO presigned or local serve)."""
    p         = await _get_person_or_404(db, person_id)
    face_repo = _get_face_repo(request)
    cid       = str(p.client_id)

    import asyncio, datetime
    images = await face_repo.get_enrollment_images(cid, person_id)
    minio  = request.app.state.minio

    # Collect image_refs from ALL active embedding rows (one row per template)
    all_embs = (await db.execute(
        select(FaceEmbedding)
        .where(
            FaceEmbedding.person_id == uuid.UUID(person_id),
            FaceEmbedding.is_active.is_(True),
        )
        .order_by(FaceEmbedding.created_at.asc())
    )).scalars().all()

    # Build list of (ref, bucket) pairs — LIVE_ASSIGNMENT images are in face-evidence
    evidence_bucket   = getattr(request.app.state.settings, "minio_evidence_bucket",  "face-evidence")
    enrollment_bucket = getattr(request.app.state.settings, "minio_enrollment_bucket", "face-enrollment")
    ref_bucket_pairs: list[tuple[str, str]] = []
    for emb in all_embs:
        bucket = evidence_bucket if emb.source == "LIVE_ASSIGNMENT" else enrollment_bucket
        if emb.image_refs:
            for ref in emb.image_refs:
                ref_bucket_pairs.append((ref, bucket))

    urls: list[str] = []

    # Build URLs from image_refs (more reliable than index-based guessing)
    for ref, bucket in ref_bucket_pairs:
        if ref.startswith("local:"):
            # Serve via our own endpoint
            raw_key = ref[6:]  # strip "local:"
            urls.append(f"/api/enrollment/image-file/{raw_key}")
        else:
            try:
                url = await asyncio.to_thread(
                    minio.presigned_get_object,
                    bucket,
                    ref,
                    expires=datetime.timedelta(hours=1),
                )
                urls.append(url)
            except Exception:
                pass

    # If no refs but we have raw image bytes, generate data URIs
    if not urls and images:
        import base64
        for img_bytes in images:
            b64 = base64.b64encode(img_bytes).decode()
            urls.append(f"data:image/jpeg;base64,{b64}")

    items = [
        {"url": u, "image_id": f"{person_id}_{i}", "quality_score": None, "created_at": None}
        for i, u in enumerate(urls)
    ]
    return {
        "person_id": person_id,
        "count":     len(urls),
        "urls":      urls,
        "items":     items,
    }


@router.put("/{person_id}/re-enroll", dependencies=[_WR])
async def re_enroll(
    person_id: str,
    body:      EnrollRequest,
    request:   Request,
    db:        DBSession,
) -> dict:
    """
    Replace all existing face embeddings with a fresh enrollment.
    Deactivates old embeddings before creating new ones.
    """
    cid       = resolve_client_id(request, body.client_id, require=False)
    # SUPER_ADMIN without explicit client_id — derive from the person record
    if not cid:
        row = (await db.execute(
            text("SELECT client_id::text FROM persons WHERE person_id = (:pid)::uuid LIMIT 1"),
            {"pid": person_id},
        )).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Person not found")
        cid = row[0]
    face_repo = _get_face_repo(request)
    minio     = request.app.state.minio

    import asyncio, base64

    _re_enroll_log = logging.getLogger(__name__)
    _re_enroll_log.info(
        "re_enroll: person=%s keys_count=%d key_prefixes=%s",
        person_id,
        len(body.image_keys),
        [k[:20] for k in body.image_keys],
    )

    async def _get_img(key: str) -> bytes | None:
        if key.startswith("b64:"):
            try:
                data = base64.b64decode(key[4:])
                _re_enroll_log.info("re_enroll _get_img: b64 decoded %d bytes from key len=%d", len(data), len(key))
                return data if data else None
            except Exception as exc:
                _re_enroll_log.error("re_enroll _get_img: b64 decode failed: %s (key len=%d, prefix=%s)", exc, len(key), key[:30])
                return None
        try:
            evidence_bucket = getattr(request.app.state.settings, "minio_evidence_bucket", "face-evidence")
            return await asyncio.to_thread(
                lambda: minio.get_object(evidence_bucket, key).read()
            )
        except Exception as exc:
            _re_enroll_log.error("re_enroll _get_img: minio fetch failed for key=%s: %s", key, exc)
            return None

    images = [img for img in await asyncio.gather(*[_get_img(k) for k in body.image_keys]) if img]
    _re_enroll_log.info("re_enroll: decoded %d/%d images", len(images), len(body.image_keys))
    if not images:
        raise HTTPException(status_code=400, detail="No valid images found")

    result = await face_repo.enroll_person(cid, person_id, images, {})
    if result.error:
        raise HTTPException(status_code=422, detail=result.error)

    audit(request, "RE_ENROLL", "person", person_id)
    return {
        "person_id":     result.person_id,
        "embedding_id":  result.embedding_id,
        "images_passed": result.images_passed,
    }


@router.delete("/{person_id}", status_code=204, dependencies=[_DEL])
async def delete_enrollment(person_id: str, request: Request, db: DBSession) -> None:
    """Deactivate all face embeddings and remove MinIO images for a person."""
    await _get_person_or_404(db, person_id)
    face_repo = _get_face_repo(request)
    cid       = resolve_client_id(request)
    await face_repo.delete_person(cid, person_id)
    audit(request, "DELETE_ENROLLMENT", "person", person_id)
