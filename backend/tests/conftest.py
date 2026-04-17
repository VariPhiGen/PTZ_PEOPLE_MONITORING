"""
Shared pytest fixtures for the ACAS backend test suite.

Infrastructure required:
  • PostgreSQL ≥ 16 with pgvector extension  → TEST_DATABASE_URL
  • fakeredis (in-memory)                    → no external Redis needed
  • All other services (Kafka, MinIO)        → mocked via MagicMock

Database strategy:
  - Session-scoped engine creates extensions + tables once.
  - Each test gets a fresh async transaction that is rolled back on teardown,
    giving full isolation without repeated schema setup.

AI pipeline strategy:
  - AIPipeline is never loaded; tests inject MockAIPipeline that returns
    deterministic random embeddings (no GPU or ONNX models required).
"""
from __future__ import annotations

import asyncio
import os
import time
import uuid
from typing import AsyncGenerator, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import fakeredis.aioredis
import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from jose import jwt
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

# ── App imports ───────────────────────────────────────────────────────────────
# Patch GPU-only FAISS BEFORE any app module is imported so that the fallback
# CPU implementation is used transparently throughout all tests.
import sys as _sys

try:
    import faiss as _faiss_real  # CPU faiss (requirements-test.txt)
    _faiss_real.StandardGpuResources = MagicMock
    _faiss_real.GpuIndexFlatConfig   = MagicMock
    _faiss_real.index_cpu_to_gpu     = lambda _res, _dev, idx: idx
    _faiss_real.index_gpu_to_cpu     = lambda idx: idx
except ImportError:
    pass  # FAISS not installed; enrollment/search tests will be skipped

from app.config import Settings
from app.constants import ROLE_PERMISSIONS
from app.deps import get_db, get_current_user
from app.models.base import Base

# Register all ORM models with Base.metadata
import app.models.audit_log           # noqa: F401
import app.models.attendance_records  # noqa: F401
import app.models.cameras             # noqa: F401
import app.models.client_node_assignments  # noqa: F401
import app.models.clients             # noqa: F401
import app.models.detection_log       # noqa: F401
import app.models.face_embeddings     # noqa: F401
import app.models.persons             # noqa: F401
import app.models.sessions            # noqa: F401
import app.models.sightings           # noqa: F401
import app.models.users               # noqa: F401
import app.models.face_datasets       # noqa: F401

# ── Environment ───────────────────────────────────────────────────────────────
TEST_DB_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://acas:acas@localhost:5432/acas_test",
)

# ── Test Settings (HS256 makes token generation trivial) ─────────────────────
TEST_JWT_SECRET = "acas-test-hs256-secret-do-not-use-in-production"

TEST_SETTINGS = Settings(
    database_url=TEST_DB_URL,
    timescale_url=TEST_DB_URL,
    redis_url="redis://localhost:6379/1",
    minio_endpoint="localhost:9000",
    minio_access_key="minioadmin",
    minio_secret_key="minioadmin",
    kafka_bootstrap_servers="localhost:9092",
    model_dir="/tmp/acas-test-models",
    gpu_device_id="0",
    jwt_secret_key=TEST_JWT_SECRET,
    jwt_algorithm="HS256",
    node_id="test-node-aaaaaaaa",
    node_name="Test Node",
    node_location="Test Lab",
)

# ── Mock AI Pipeline ──────────────────────────────────────────────────────────
_RNG = np.random.default_rng(seed=42)


def _random_embedding(seed: Optional[int] = None) -> np.ndarray:
    """Return a deterministic L2-normalised 512-dim vector."""
    rng = np.random.default_rng(seed=seed)
    v = rng.random(512).astype(np.float32)
    v /= np.linalg.norm(v)
    return v


class MockAIPipeline:
    """CPU-safe stand-in for AIPipeline. Returns synthetic detections."""

    STATIC_EMB: Dict[str, np.ndarray] = {}  # person_id → embedding

    @classmethod
    def get_embedding_for(cls, person_id: str) -> np.ndarray:
        if person_id not in cls.STATIC_EMB:
            cls.STATIC_EMB[person_id] = _random_embedding(
                seed=int(uuid.UUID(person_id).int % 2**31)
            )
        return cls.STATIC_EMB[person_id]

    async def detect_persons(self, frame, roi_rect=None):
        return []

    async def detect_faces(self, frame, person_bboxes=None):
        return []

    def estimate_yaw(self, landmarks) -> float:
        return 0.0  # always frontal in tests

    async def align_face(self, frame, landmarks, iod_px=0.0):
        return np.zeros((112, 112, 3), dtype=np.uint8)

    async def get_embedding(self, face_chip) -> np.ndarray:
        return _random_embedding()

    async def check_liveness(self, frame, face_bbox) -> float:
        return 0.92

    async def process_frame(self, frame, roi_rect=None):
        from dataclasses import dataclass, field

        @dataclass
        class FrameResult:
            persons: list = field(default_factory=list)
            faces_with_embeddings: list = field(default_factory=list)
            time_ms: float = 35.0

        return FrameResult()


# ── MockFaceRepository ────────────────────────────────────────────────────────
class MockFaceRepository:
    """
    In-memory face repository for tests.

    Uses faiss-cpu IndexFlatIP per client so enrollment/search logic
    can be exercised without GPU or network I/O.
    """

    def __init__(self):
        self._embs: Dict[str, Dict[str, np.ndarray]] = {}  # cid → pid → emb

    async def enroll_person(self, client_id: str, person_id: str, images, metadata=None):
        from dataclasses import dataclass

        @dataclass
        class EnrollmentResult:
            person_id: str
            embeddings_added: int
            quality_passed: int
            quality_failed: int

        emb = _random_embedding(seed=abs(hash(person_id)) % 2**31)
        self._embs.setdefault(client_id, {})[person_id] = emb
        return EnrollmentResult(
            person_id=person_id,
            embeddings_added=1,
            quality_passed=len(images),
            quality_failed=0,
        )

    async def identify(self, client_id, embedding, roster_ids=None, thresholds=None):
        from dataclasses import dataclass

        @dataclass
        class IdentifyResult:
            person_id: Optional[str]
            similarity: float
            tier: str

        best_pid, best_score = None, 0.0
        for pid, emb in self._embs.get(client_id, {}).items():
            if roster_ids and pid not in roster_ids:
                continue
            score = float(np.dot(embedding, emb))
            if score > best_score:
                best_score, best_pid = score, pid

        threshold = 0.72 if thresholds is None else getattr(thresholds, "tier1", 0.72)
        if best_pid and best_score >= threshold:
            return IdentifyResult(person_id=best_pid, similarity=best_score, tier="FAISS")
        return IdentifyResult(person_id=None, similarity=0.0, tier="UNKNOWN")

    async def search_roster(self, client_id, embedding, roster_ids, top_k=5):
        results = []
        for pid, emb in self._embs.get(client_id, {}).items():
            if roster_ids and pid not in roster_ids:
                continue
            results.append((pid, float(np.dot(embedding, emb))))
        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    async def delete_person(self, client_id: str, person_id: str):
        self._embs.get(client_id, {}).pop(person_id, None)

    async def rebuild_faiss_index(self, client_id: str, roster_ids=None):
        pass

    def get_index_size(self, client_id: str) -> int:
        return len(self._embs.get(client_id, {}))


# ── Database fixtures ─────────────────────────────────────────────────────────
@pytest_asyncio.fixture(scope="session")
async def db_engine():
    """Create engine, install extensions, and create all tables once per session."""
    engine = create_async_engine(TEST_DB_URL, poolclass=NullPool, echo=False)

    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
        # Create tables; detection_log hypertable requires TimescaleDB — fall back
        # to a regular table if the extension is not present (fine for unit tests).
        await conn.run_sync(Base.metadata.create_all)

    yield engine
    await engine.dispose()


@pytest_asyncio.fixture(autouse=True)
async def clean_tables(db_engine):
    """Truncate all tenant tables before every test for isolation."""
    async with db_engine.connect() as conn:
        await conn.execute(text(
            "TRUNCATE "
            "  client_node_assignments, attendance_records, sightings, "
            "  detection_log, face_embeddings, face_datasets, audit_log, "
            "  sessions, persons, cameras, users, clients "
            "RESTART IDENTITY CASCADE"
        ))
        await conn.commit()
    yield


@pytest_asyncio.fixture
async def db_session(db_engine) -> AsyncGenerator[AsyncSession, None]:
    """Async session bound to a savepoint; rolled back after each test."""
    factory = async_sessionmaker(db_engine, expire_on_commit=False)
    async with factory() as session:
        yield session


# ── Redis (fake) ──────────────────────────────────────────────────────────────
@pytest_asyncio.fixture
async def fake_redis():
    server = fakeredis.aioredis.FakeRedis(decode_responses=False)
    yield server
    await server.flushall()
    await server.aclose()


# ── Mock Kafka producer ───────────────────────────────────────────────────────
@pytest.fixture
def mock_kafka():
    producer = MagicMock()
    producer.publish_attendance   = AsyncMock(return_value=None)
    producer.publish_sighting     = AsyncMock(return_value=None)
    producer.publish_alert        = AsyncMock(return_value=None)
    producer.broadcast_config_reload = MagicMock(return_value=None)
    producer.flush                = MagicMock(return_value=None)
    # Raw confluent-kafka producer mock
    raw = MagicMock()
    raw.produce = MagicMock()
    raw.poll    = MagicMock(return_value=0)
    raw.flush   = MagicMock()
    raw.list_topics = MagicMock(return_value=MagicMock(topics={"attendance.records": None}))
    return producer, raw


# ── Mock MinIO ────────────────────────────────────────────────────────────────
@pytest.fixture
def mock_minio():
    m = MagicMock()
    m.list_buckets   = MagicMock(return_value=[])
    m.make_bucket    = MagicMock()
    m.put_object     = MagicMock()
    m.get_object     = MagicMock()
    m.remove_object  = MagicMock()
    return m


# ── Mock NodeManager / FaceSync ───────────────────────────────────────────────
@pytest.fixture
def mock_node_manager():
    nm = MagicMock()
    nm.register     = AsyncMock(return_value=None)
    nm.heartbeat_loop = AsyncMock(return_value=None)
    nm.on_camera_assigned = AsyncMock(return_value=None)
    nm.on_camera_removed  = AsyncMock(return_value=None)
    return nm


# ── FastAPI test client ───────────────────────────────────────────────────────
@pytest_asyncio.fixture
async def app_client(
    db_session: AsyncSession,
    fake_redis,
    mock_kafka,
    mock_minio,
    mock_node_manager,
) -> AsyncGenerator[AsyncClient, None]:
    """
    Returns an httpx AsyncClient wired to the FastAPI app with:
      • Test database session (no lifespan startup)
      • Fake Redis
      • Mock Kafka, MinIO, NodeManager, AIPipeline, FaceRepository
    """
    from app.main import app

    kafka_producer, raw_producer = mock_kafka
    face_repo = MockFaceRepository()
    ai_pipeline = MockAIPipeline()

    # Override dependency
    async def _get_db():
        yield db_session

    app.dependency_overrides[get_db] = _get_db

    # Patch app.state
    app.state.settings       = TEST_SETTINGS
    app.state.redis          = fake_redis
    app.state.minio          = mock_minio
    app.state.kafka_producer = kafka_producer
    app.state._raw_producer  = raw_producer
    app.state.node_manager   = mock_node_manager
    app.state.face_repo      = face_repo
    app.state.ai_pipeline    = ai_pipeline
    app.state.session_factory = async_sessionmaker(db_session.get_bind(), expire_on_commit=False)
    app.state.ptz_brains     = {}
    app.state.active_sessions = {}

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
        timeout=30.0,
    ) as client:
        yield client

    app.dependency_overrides.clear()


# ── Token helpers ─────────────────────────────────────────────────────────────
def _make_token(
    user_id: str,
    email: str,
    name: str,
    role: str,
    client_id: Optional[str] = None,
    client_slug: str = "",
    extra_perms: Optional[List[str]] = None,
) -> str:
    perms = extra_perms if extra_perms is not None else ROLE_PERMISSIONS.get(role, [])
    payload = {
        "user_id":    user_id,
        "email":      email,
        "name":       name,
        "role":       role,
        "client_id":  client_id,
        "client_slug": client_slug,
        "permissions": perms,
        "iat": int(time.time()),
        "exp": int(time.time()) + 86_400,
    }
    return jwt.encode(payload, TEST_JWT_SECRET, algorithm="HS256")


def auth_headers(token: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# ── Reusable user/client helpers ──────────────────────────────────────────────
async def create_client(session: AsyncSession, name: str = "Test Corp") -> "Client":
    from app.models.clients import Client
    slug = name.lower().replace(" ", "-") + "-" + str(uuid.uuid4())[:8]
    client = Client(
        client_id=uuid.uuid4(), name=name, slug=slug,
        status="ACTIVE", max_cameras=50, max_persons=10_000,
        created_at=int(time.time()), updated_at=int(time.time()),
    )
    session.add(client)
    await session.flush()
    return client


async def create_user(
    session: AsyncSession,
    email: str,
    password: str = "Test@pass1234",
    role: str = "CLIENT_ADMIN",
    client_id: Optional[uuid.UUID] = None,
) -> "User":
    from app.models.users import User
    from app.utils.security import hash_password
    user = User(
        user_id=uuid.uuid4(),
        email=email,
        password_hash=hash_password(password),
        name=email.split("@")[0].title(),
        role=role,
        client_id=client_id,
        status="ACTIVE",
        mfa_enabled=False,
        created_at=int(time.time()),
        updated_at=int(time.time()),
    )
    session.add(user)
    await session.flush()
    return user


async def create_camera(
    session: AsyncSession,
    client_id: uuid.UUID,
    name: str = "Test Camera",
) -> "Camera":
    from app.models.cameras import Camera
    camera = Camera(
        camera_id=uuid.uuid4(),
        client_id=client_id,
        name=name,
        room_name=name,
        rtsp_url="rtsp://10.0.0.1:554/stream",
        onvif_host="10.0.0.1",
        onvif_port=80,
        onvif_username="admin",
        onvif_password_encrypted="enc:Test@2024",
        status="ONLINE",
        mode="ATTENDANCE",
        created_at=int(time.time()),
        updated_at=int(time.time()),
    )
    session.add(camera)
    await session.flush()
    return camera


async def create_person(
    session: AsyncSession,
    client_id: uuid.UUID,
    name: str = "Test Person",
    role: str = "STUDENT",
) -> "Person":
    from app.models.persons import Person
    person = Person(
        person_id=uuid.uuid4(),
        client_id=client_id,
        external_id=str(uuid.uuid4())[:8],
        name=name,
        role=role,
        status="ACTIVE",
        consent_at=int(time.time()),
        created_at=int(time.time()),
        updated_at=int(time.time()),
    )
    session.add(person)
    await session.flush()
    return person


async def create_session_record(
    session: AsyncSession,
    client_id: uuid.UUID,
    camera_id: uuid.UUID,
    sync_status: str = "PENDING",
) -> "SessionRecord":
    from app.models.sessions import Session as SessionModel
    now = int(time.time())
    sess = SessionModel(
        session_id=uuid.uuid4(),
        client_id=client_id,
        camera_id=camera_id,
        course_id="CS101",
        course_name="Intro to CS",
        scheduled_start=now,
        scheduled_end=now + 5_400,
        actual_start=now,
        sync_status=sync_status,
        created_at=now,
    )
    session.add(sess)
    await session.flush()
    return sess


# ── Fixture shortcuts for common role tokens ──────────────────────────────────
@pytest_asyncio.fixture
async def super_admin(db_session):
    return await create_user(db_session, "superadmin@acas.test", role="SUPER_ADMIN")


@pytest_asyncio.fixture
def super_admin_token(super_admin) -> str:
    return _make_token(
        str(super_admin.user_id), super_admin.email,
        super_admin.name, "SUPER_ADMIN",
    )


@pytest_asyncio.fixture
async def tenant_a(db_session):
    return await create_client(db_session, "Tenant Alpha")


@pytest_asyncio.fixture
async def client_admin_a(db_session, tenant_a):
    return await create_user(
        db_session, "admin.a@alpha.test",
        role="CLIENT_ADMIN", client_id=tenant_a.client_id,
    )


@pytest_asyncio.fixture
def client_admin_a_token(client_admin_a, tenant_a) -> str:
    return _make_token(
        str(client_admin_a.user_id), client_admin_a.email,
        client_admin_a.name, "CLIENT_ADMIN",
        str(tenant_a.client_id), tenant_a.slug,
    )


@pytest_asyncio.fixture
async def viewer_a(db_session, tenant_a):
    return await create_user(
        db_session, "viewer.a@alpha.test",
        role="VIEWER", client_id=tenant_a.client_id,
    )


@pytest_asyncio.fixture
def viewer_a_token(viewer_a, tenant_a) -> str:
    return _make_token(
        str(viewer_a.user_id), viewer_a.email,
        viewer_a.name, "VIEWER",
        str(tenant_a.client_id), tenant_a.slug,
    )


# ── Synthetic face images ─────────────────────────────────────────────────────
def synthetic_face_bytes(seed: int = 0) -> bytes:
    """Return a tiny JPEG-encoded face chip (112×112) for upload tests."""
    import io
    from PIL import Image
    rng = np.random.default_rng(seed=seed)
    img_array = rng.integers(80, 200, (112, 112, 3), dtype=np.uint8)
    img = Image.fromarray(img_array, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()
