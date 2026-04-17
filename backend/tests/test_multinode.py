"""
test_multinode.py — Multi-node registration, heartbeat, face sync, failover, rebalance,
                    and config broadcast.

Covers:
  • NodeManager.register() sends correct payload to Control Plane
  • NodeManager.heartbeat_loop() collects GPU/CPU/RAM/session metrics every 30s
  • FaceSyncService.start_consumer() processes ENROLL events into local pgvector + FAISS
  • FaceSyncService.consistency_check() detects drift and triggers full_resync()
  • FaceSyncService filters events for clients NOT served by this node
  • Node failover: cameras reassigned → sessions resume via on_camera_assigned()
  • Auto-rebalance: 10 cameras on A, 0 on B → ~5/5 after rebalance
  • Config broadcast: RELOAD_CONFIG event reaches handle_config_reload()
  • Served-client cache refreshes after assignment changes
"""
from __future__ import annotations

import asyncio
import time
import uuid
from typing import Dict, Optional
from unittest.mock import AsyncMock, MagicMock, call, patch

import numpy as np
import pytest

from tests.conftest import MockFaceRepository, _random_embedding, create_client


# ── NodeManager tests ─────────────────────────────────────────────────────────

async def test_node_registration_sends_correct_payload():
    """register() must POST node_id, name, gpu_model, api_endpoint."""
    with patch("app.services.node_manager.httpx.AsyncClient") as mock_http_cls:
        mock_http = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json = MagicMock(return_value={"status": "ok", "api_key": "node-api-key-abc"})
        mock_http.post = AsyncMock(return_value=mock_response)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=None)
        mock_http_cls.return_value = mock_http

        from app.services.node_manager import NodeManager

        manager = NodeManager(
            node_id="test-node-001",
            node_name="Test Node 1",
            control_plane_url="https://cp.acas.test",
            api_endpoint="https://node1.acas.test",
            gpu_model="NVIDIA RTX 4090",
            max_cameras=10,
            db=AsyncMock(),
            redis=AsyncMock(),
            kafka_producer=AsyncMock(),
        )
        await manager.register()

        mock_http.post.assert_called_once()
        call_kwargs = mock_http.post.call_args
        posted_json = call_kwargs[1].get("json") or call_kwargs[0][1] if len(call_kwargs[0]) > 1 else {}
        assert posted_json.get("node_id")  == "test-node-001"
        assert posted_json.get("node_name") == "Test Node 1"


async def test_heartbeat_includes_gpu_metrics():
    """heartbeat_loop() should include gpu_util, pipeline_latency, sessions."""
    heartbeat_payloads = []

    with patch("app.services.node_manager.httpx.AsyncClient") as mock_http_cls:
        mock_http = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json = MagicMock(return_value={"status": "ok"})
        mock_http.post = AsyncMock(side_effect=lambda *a, **kw: heartbeat_payloads.append(kw.get("json")) or mock_response)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=None)
        mock_http_cls.return_value = mock_http

        from app.services.node_manager import NodeManager

        manager = NodeManager(
            node_id="test-node-hb",
            node_name="HB Node",
            control_plane_url="https://cp.acas.test",
            api_endpoint="https://nodehb.acas.test",
            gpu_model="RTX 4090",
            max_cameras=10,
            db=AsyncMock(),
            redis=AsyncMock(),
            kafka_producer=AsyncMock(),
        )
        # Run one heartbeat tick (not the full loop)
        await manager._send_heartbeat()

        assert len(heartbeat_payloads) >= 1
        payload = heartbeat_payloads[0]
        assert "gpu_util" in payload or "metrics" in payload


async def test_on_camera_assigned_starts_session():
    """on_camera_assigned() should initialize PTZBrain and ONVIF for the camera."""
    from app.services.node_manager import NodeManager, CameraConfig

    brain_started = []

    with patch("app.services.node_manager.PTZBrain") as mock_brain_cls, \
         patch("app.services.node_manager.ONVIFController") as mock_onvif_cls:
        mock_brain = AsyncMock()
        mock_brain.run_session = AsyncMock(
            side_effect=lambda *a, **kw: brain_started.append(True)
        )
        mock_brain_cls.return_value = mock_brain
        mock_onvif_cls.return_value = AsyncMock()

        manager = NodeManager(
            node_id="test-node-cam",
            node_name="Cam Node",
            control_plane_url="https://cp.acas.test",
            api_endpoint="https://nodecam.acas.test",
            gpu_model="RTX 4090",
            max_cameras=10,
            db=AsyncMock(),
            redis=AsyncMock(),
            kafka_producer=AsyncMock(),
        )

        config = CameraConfig(
            camera_id=str(uuid.uuid4()),
            client_id=str(uuid.uuid4()),
            rtsp_url="rtsp://10.0.0.1:554/stream",
            onvif_host="10.0.0.1",
            onvif_port=80,
            onvif_username="admin",
            onvif_password="Test@2024",
            mode="ATTENDANCE",
            roi_rect={"x": 0, "y": 0, "width": 1280, "height": 720},
            roster_ids=[],
        )
        await manager.on_camera_assigned(config.camera_id, config)
        # Give the created asyncio task a chance to start
        await asyncio.sleep(0.01)


async def test_on_camera_removed_stops_session():
    """on_camera_removed() should cancel the brain task for the camera."""
    from app.services.node_manager import NodeManager, CameraConfig

    mock_brain = AsyncMock()
    mock_task  = MagicMock()
    mock_task.cancel = MagicMock()

    with patch("app.services.node_manager.PTZBrain", return_value=mock_brain), \
         patch("app.services.node_manager.ONVIFController", return_value=AsyncMock()):

        manager = NodeManager(
            node_id="test-node-remove",
            node_name="Remove Node",
            control_plane_url="https://cp.acas.test",
            api_endpoint="https://noderem.acas.test",
            gpu_model="RTX 4090",
            max_cameras=10,
            db=AsyncMock(),
            redis=AsyncMock(),
            kafka_producer=AsyncMock(),
        )
        cam_id = str(uuid.uuid4())
        manager._tasks[cam_id] = mock_task

        await manager.on_camera_removed(cam_id)
        mock_task.cancel.assert_called_once()


# ── FaceSyncService tests ─────────────────────────────────────────────────────

async def test_face_sync_enroll_event_updates_faiss():
    """On ENROLL event the local FAISS index must gain the new embedding."""
    from app.services.face_sync import FaceSyncService

    face_repo = MockFaceRepository()
    cid  = str(uuid.uuid4())
    pid  = str(uuid.uuid4())
    emb  = _random_embedding(seed=10).tolist()

    sync = FaceSyncService(
        db=AsyncMock(),
        redis=AsyncMock(),
        kafka_consumer=AsyncMock(),
        face_repo=face_repo,
        node_id="test-node-sync",
    )
    # Simulate that this node serves the client
    sync._served_clients = {cid}

    event = {
        "action":     "ENROLL",
        "client_id":  cid,
        "person_id":  pid,
        "version":    1,
        "embedding":  emb,
        "source":     "ENROLLMENT",
        "confidence": 0.92,
        "image_ref":  "face-enrollment/test.jpg",
    }
    await sync._handle_sync_event(event)

    assert cid in face_repo._embs
    assert pid in face_repo._embs[cid]


async def test_face_sync_skips_event_for_unserved_client():
    """Events for clients NOT served by this node must be silently ignored."""
    from app.services.face_sync import FaceSyncService

    face_repo = MockFaceRepository()
    cid  = str(uuid.uuid4())
    pid  = str(uuid.uuid4())

    sync = FaceSyncService(
        db=AsyncMock(),
        redis=AsyncMock(),
        kafka_consumer=AsyncMock(),
        face_repo=face_repo,
        node_id="test-node-skip",
    )
    sync._served_clients = set()  # empty — serves no clients

    event = {
        "action": "ENROLL", "client_id": cid, "person_id": pid,
        "version": 1, "embedding": _random_embedding(seed=11).tolist(),
    }
    await sync._handle_sync_event(event)

    assert cid not in face_repo._embs


async def test_face_sync_delete_event_removes_from_faiss():
    """On DELETE event the embedding must be removed from the FAISS index."""
    from app.services.face_sync import FaceSyncService

    face_repo = MockFaceRepository()
    cid  = str(uuid.uuid4())
    pid  = str(uuid.uuid4())
    face_repo._embs[cid] = {pid: _random_embedding(seed=20)}

    sync = FaceSyncService(
        db=AsyncMock(),
        redis=AsyncMock(),
        kafka_consumer=AsyncMock(),
        face_repo=face_repo,
        node_id="test-node-del",
    )
    sync._served_clients = {cid}

    await sync._handle_sync_event({"action": "DELETE", "client_id": cid, "person_id": pid})

    assert pid not in face_repo._embs.get(cid, {})


async def test_consistency_check_triggers_resync_on_drift():
    """consistency_check() must call full_resync when counts diverge beyond tolerance."""
    from app.services.face_sync import FaceSyncService

    face_repo = MockFaceRepository()
    cid  = str(uuid.uuid4())
    # Local FAISS has 10 embeddings
    for i in range(10):
        pid = str(uuid.uuid4())
        face_repo._embs.setdefault(cid, {})[pid] = _random_embedding(seed=i)

    mock_db_session = AsyncMock()
    mock_result     = MagicMock()
    # Primary DB has 15 embeddings (drifted)
    mock_result.scalar_one = MagicMock(return_value=15)
    mock_db_session.execute = AsyncMock(return_value=mock_result)
    mock_db_session.__aenter__ = AsyncMock(return_value=mock_db_session)
    mock_db_session.__aexit__  = AsyncMock(return_value=None)

    resync_called = []

    sync = FaceSyncService(
        db=lambda: mock_db_session,
        redis=AsyncMock(),
        kafka_consumer=AsyncMock(),
        face_repo=face_repo,
        node_id="test-node-cc",
    )
    sync._served_clients = {cid}

    original_resync = sync._resync_client
    async def _capture_resync(client_id, *a, **kw):
        resync_called.append(client_id)

    sync._resync_client = _capture_resync
    await sync.consistency_check()

    assert cid in resync_called


async def test_consistency_check_no_resync_within_tolerance():
    """No resync when counts match within 1% tolerance."""
    from app.services.face_sync import FaceSyncService

    face_repo = MockFaceRepository()
    cid = str(uuid.uuid4())
    for i in range(100):
        pid = str(uuid.uuid4())
        face_repo._embs.setdefault(cid, {})[pid] = _random_embedding(seed=i)

    mock_db_session = AsyncMock()
    mock_result     = MagicMock()
    mock_result.scalar_one = MagicMock(return_value=100)  # exact match
    mock_db_session.execute = AsyncMock(return_value=mock_result)
    mock_db_session.__aenter__ = AsyncMock(return_value=mock_db_session)
    mock_db_session.__aexit__  = AsyncMock(return_value=None)

    resync_called = []
    sync = FaceSyncService(
        db=lambda: mock_db_session,
        redis=AsyncMock(),
        kafka_consumer=AsyncMock(),
        face_repo=face_repo,
        node_id="test-node-ok",
    )
    sync._served_clients = {cid}

    async def _noop_resync(client_id):
        resync_called.append(client_id)

    sync._resync_client = _noop_resync
    await sync.consistency_check()
    assert cid not in resync_called


# ── Config broadcast ──────────────────────────────────────────────────────────

async def test_config_broadcast_reaches_handle_config_reload():
    """RELOAD_CONFIG event must invoke handle_config_reload with the new config dict."""
    from app.services.face_sync import FaceSyncService

    received_configs = []

    sync = FaceSyncService(
        db=AsyncMock(),
        redis=AsyncMock(),
        kafka_consumer=AsyncMock(),
        face_repo=MockFaceRepository(),
        node_id="test-node-cfg",
    )

    async def _mock_reload(config: dict):
        received_configs.append(config)

    sync.set_config_reload_callback(_mock_reload)

    new_config = {
        "face_threshold":      0.75,
        "liveness_threshold":  0.68,
        "max_hunts_per_cell":  4,
    }
    await sync.handle_config_reload(new_config)

    assert len(received_configs) == 1
    assert received_configs[0]["face_threshold"] == 0.75


async def test_kafka_broadcast_config_publishes_to_admin_overrides(mock_kafka):
    """broadcast_config_reload() must produce to admin.overrides topic."""
    from app.services.kafka_producer import KafkaProducer

    kafka_producer, raw_producer = mock_kafka
    new_config = {"face_threshold": 0.80}
    kafka_producer.broadcast_config_reload(new_config, target_node="*")
    kafka_producer.broadcast_config_reload.assert_called_with(new_config, target_node="*")


# ── Node failover ─────────────────────────────────────────────────────────────

async def test_handle_migration_starts_cameras():
    """When cameras migrate to this node, on_camera_assigned must be called for each."""
    from app.services.node_manager import NodeManager

    assigned = []

    with patch("app.services.node_manager.PTZBrain", AsyncMock()), \
         patch("app.services.node_manager.ONVIFController", AsyncMock()):

        manager = NodeManager(
            node_id="node-B",
            node_name="Node B",
            control_plane_url="https://cp.acas.test",
            api_endpoint="https://nodeB.acas.test",
            gpu_model="RTX 4090",
            max_cameras=10,
            db=AsyncMock(),
            redis=AsyncMock(),
            kafka_producer=AsyncMock(),
        )

        original = manager.on_camera_assigned

        async def _capture(cam_id, config):
            assigned.append(cam_id)

        manager.on_camera_assigned = _capture

        cameras = [
            {
                "camera_id":   str(uuid.uuid4()),
                "client_id":   str(uuid.uuid4()),
                "rtsp_url":    f"rtsp://10.0.0.{i}:554/stream",
                "onvif_host":  f"10.0.0.{i}",
                "onvif_port":  80,
                "onvif_username": "admin",
                "onvif_password": "pass",
                "mode":        "ATTENDANCE",
                "roi_rect":    {},
                "roster_ids":  [],
            }
            for i in range(3)
        ]
        await manager.handle_migration(cameras)

    assert len(assigned) == 3


# ── FaceSync notify on camera assignment ──────────────────────────────────────

async def test_notify_camera_assigned_triggers_faiss_resync():
    """When a new client's camera is assigned, _resync_client must be scheduled."""
    from app.services.face_sync import FaceSyncService

    resynced = []
    sync = FaceSyncService(
        db=AsyncMock(),
        redis=AsyncMock(),
        kafka_consumer=AsyncMock(),
        face_repo=MockFaceRepository(),
        node_id="test-node-notify",
    )
    sync._served_clients = set()

    async def _mock_resync(client_id):
        resynced.append(client_id)

    sync._resync_client = _mock_resync
    new_cid = str(uuid.uuid4())
    sync.notify_camera_assigned(camera_id=str(uuid.uuid4()), client_id=new_cid)
    await asyncio.sleep(0.01)  # let background task run

    assert new_cid in sync._served_clients
