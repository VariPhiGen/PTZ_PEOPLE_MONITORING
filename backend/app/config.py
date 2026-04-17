from __future__ import annotations

import uuid
from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Databases
    database_url: str = Field(..., validation_alias="DATABASE_URL")
    timescale_url: str = Field(..., validation_alias="TIMESCALE_URL")
    redis_url: str = Field(..., validation_alias="REDIS_URL")

    # Object storage
    minio_endpoint: str = Field(..., validation_alias="MINIO_ENDPOINT")
    minio_access_key: str = Field(..., validation_alias="MINIO_ACCESS_KEY")
    minio_secret_key: str = Field(..., validation_alias="MINIO_SECRET_KEY")

    # Messaging — local cluster (high-frequency intra-node events)
    kafka_bootstrap_servers: str = Field(..., validation_alias="KAFKA_BOOTSTRAP_SERVERS")
    kafka_schema_registry_url: Optional[str] = Field(
        None, validation_alias="KAFKA_SCHEMA_REGISTRY_URL"
    )
    # Central cluster (business events, cross-node repo.sync.embeddings)
    # Falls back to local if not set.
    central_kafka_bootstrap_servers: Optional[str] = Field(
        None, validation_alias="CENTRAL_KAFKA_BOOTSTRAP_SERVERS"
    )

    # Inference
    model_dir: str = Field(..., validation_alias="MODEL_DIR")
    gpu_device_id: str = Field("0", validation_alias="GPU_DEVICE_ID")
    gpu_max_concurrent: int = Field(3, validation_alias="GPU_MAX_CONCURRENT")
    gpu_vram_budget_gb: float = Field(20.0, validation_alias="GPU_VRAM_BUDGET_GB")

    # Auth — for RS256 put the RSA private-key PEM here; for HS256 use a shared secret.
    jwt_secret_key: str = Field(..., validation_alias="JWT_SECRET_KEY")
    jwt_algorithm: str = Field("RS256", validation_alias="JWT_ALGORITHM")

    # Control plane (optional; used when this node reports to a Cloudflare CP)
    control_plane_url: Optional[str] = Field(None, validation_alias="CONTROL_PLANE_URL")
    node_auth_token:   Optional[str] = Field(None, validation_alias="NODE_AUTH_TOKEN")

    # Dashboard URL — used to build shareable enrollment links.
    # Set to the public-facing Next.js dashboard URL (e.g. https://app.yourcompany.com).
    # If unset, enrollment links are built from the incoming request's Host header.
    dashboard_url: Optional[str] = Field(None, validation_alias="DASHBOARD_URL")

    # Node identity — NODE_ID must persist across restarts (write to .env after first boot).
    node_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        validation_alias="NODE_ID",
    )
    node_name:        str           = Field(...,   validation_alias="NODE_NAME")
    node_location:    Optional[str] = Field(None,  validation_alias="NODE_LOCATION")
    node_api_endpoint: Optional[str] = Field(None, validation_alias="NODE_API_ENDPOINT")
    gpu_model:        Optional[str] = Field(None,  validation_alias="GPU_MODEL")
    max_cameras_per_node: int       = Field(10,    validation_alias="MAX_CAMERAS_PER_NODE")

    # ── FAISS identification thresholds ────────────────────────────────────────
    faiss_tier1_threshold: float = Field(0.40, validation_alias="FAISS_TIER1_THRESHOLD")
    faiss_tier2_threshold: float = Field(0.45, validation_alias="FAISS_TIER2_THRESHOLD")

    # ── PTZ brain tuning ───────────────────────────────────────────────────────
    ptz_faculty_hunt_timeout_s: float = Field(300.0, validation_alias="PTZ_FACULTY_HUNT_TIMEOUT_S")
    ptz_monitoring_overview_s:  float = Field(300.0, validation_alias="PTZ_MONITORING_OVERVIEW_S")
    ptz_cell_settle_s:          float = Field(0.35,  validation_alias="PTZ_CELL_SETTLE_S")
    ptz_recognize_frames:       int   = Field(2,     validation_alias="PTZ_RECOGNIZE_FRAMES")
    ptz_face_hunt_budget_s:     float = Field(15.0,  validation_alias="PTZ_FACE_HUNT_BUDGET_S")
    ptz_absence_grace_s:        float = Field(30.0,  validation_alias="PTZ_ABSENCE_GRACE_S")
    ptz_camera_move_speed:      float = Field(0.5,   validation_alias="PTZ_CAMERA_MOVE_SPEED")

    # ── Sighting engine ────────────────────────────────────────────────────────
    sighting_inactive_timeout_s: float = Field(300.0, validation_alias="SIGHTING_INACTIVE_TIMEOUT_S")
    sighting_min_duration_s:     float = Field(5.0,   validation_alias="SIGHTING_MIN_DURATION_S")

    # ── Attendance thresholds ──────────────────────────────────────────────────
    attendance_present_frac: float = Field(0.70, validation_alias="ATTENDANCE_PRESENT_FRAC")
    attendance_late_frac:    float = Field(0.15, validation_alias="ATTENDANCE_LATE_FRAC")
    attendance_ee_frac:      float = Field(0.15, validation_alias="ATTENDANCE_EE_FRAC")
    attendance_absent_frac:  float = Field(0.15, validation_alias="ATTENDANCE_ABSENT_FRAC")

    # ── Storage ────────────────────────────────────────────────────────────────
    minio_enrollment_bucket: str = Field("face-enrollment", validation_alias="MINIO_ENROLLMENT_BUCKET")
    minio_evidence_bucket:   str = Field("face-evidence",   validation_alias="MINIO_EVIDENCE_BUCKET")
    enrollment_fallback_dir: str = Field("/enrollment-images", validation_alias="ENROLLMENT_FALLBACK_DIR")

    # ── Redis namespace ────────────────────────────────────────────────────────
    redis_key_prefix: str = Field("acas", validation_alias="REDIS_KEY_PREFIX")

    # ── WebSocket ──────────────────────────────────────────────────────────────
    ws_ping_interval_s: int = Field(30,  validation_alias="WS_PING_INTERVAL_S")
    ws_max_idle_s:      int = Field(300, validation_alias="WS_MAX_IDLE_S")

    # ── ONVIF password encryption (Fernet key, URL-safe base64 32 bytes) ──────
    # Generate with: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    onvif_encryption_key: Optional[str] = Field(None, validation_alias="ONVIF_ENCRYPTION_KEY")


@lru_cache
def get_settings() -> Settings:
    return Settings()
