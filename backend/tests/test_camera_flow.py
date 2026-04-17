"""
test_camera_flow.py — Camera CRUD, ONVIF mock, ROI, PTZ control.

Covers:
  • Create camera (validates fields, respects quota)
  • List, get, update, delete cameras
  • Test ONVIF connection (mocked)
  • Save ROI rect + faculty zone
  • PTZ move (absolute/relative/stop) via mock
  • PTZ status
  • Snapshot (mock returns JPEG bytes)
"""
from __future__ import annotations

import io
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from tests.conftest import (
    auth_headers,
    create_camera,
    create_client,
    create_user,
)
from tests.factories import ClientNodeAssignmentFactory


# ── Mock ONVIFController ──────────────────────────────────────────────────────

def _mock_onvif():
    ctrl = AsyncMock()
    ctrl.get_rtsp_url      = AsyncMock(return_value="rtsp://10.0.0.1:554/stream")
    ctrl.get_ptz_status    = AsyncMock(return_value={"pan": 0.0, "tilt": 0.0, "zoom": 0.0})
    ctrl.absolute_move     = AsyncMock(return_value=None)
    ctrl.relative_move     = AsyncMock(return_value=None)
    ctrl.continuous_move   = AsyncMock(return_value=None)
    ctrl.stop              = AsyncMock(return_value=None)
    ctrl.goto_home         = AsyncMock(return_value=None)
    ctrl.get_snapshot      = AsyncMock(return_value=b"\xff\xd8\xff" + b"\x00" * 100)  # tiny JPEG
    ctrl.get_fov_at_zoom   = AsyncMock(return_value=(70.0, 45.0))
    ctrl.estimate_travel_time = AsyncMock(return_value=1.2)
    return ctrl


# ── Create / List ─────────────────────────────────────────────────────────────

async def test_create_camera_success(
    app_client, db_session, tenant_a, client_admin_a, client_admin_a_token
):
    db_session.add(ClientNodeAssignmentFactory(
        client_id=tenant_a.client_id, node_id="test-node-aaaaaaaa"
    ))
    await db_session.commit()

    resp = await app_client.post(
        "/api/cameras",
        json={
            "name":             "Hall A",
            "room_name":        "Hall A",
            "building":         "East Wing",
            "floor":            "2",
            "rtsp_url":         "rtsp://10.1.1.5:554/stream",
            "onvif_host":       "10.1.1.5",
            "onvif_port":       80,
            "onvif_username":   "admin_onvif",
            "onvif_password":   "Test@2024",
            "mode":             "ATTENDANCE",
            "node_id":          "test-node-aaaaaaaa",
        },
        headers=auth_headers(client_admin_a_token),
    )
    assert resp.status_code in (200, 201)
    data = resp.json()
    assert data["name"] == "Hall A"
    assert "camera_id" in data


async def test_create_camera_missing_required_field(
    app_client, db_session, tenant_a, client_admin_a, client_admin_a_token
):
    db_session.add(ClientNodeAssignmentFactory(
        client_id=tenant_a.client_id, node_id="test-node-aaaaaaaa"
    ))
    await db_session.commit()

    resp = await app_client.post(
        "/api/cameras",
        json={"name": "No RTSP"},   # missing rtsp_url
        headers=auth_headers(client_admin_a_token),
    )
    assert resp.status_code == 422


async def test_list_cameras_returns_own(
    app_client, db_session, tenant_a, client_admin_a, client_admin_a_token
):
    await create_camera(db_session, tenant_a.client_id, "ListCam1")
    await create_camera(db_session, tenant_a.client_id, "ListCam2")
    await db_session.commit()

    resp = await app_client.get("/api/cameras", headers=auth_headers(client_admin_a_token))
    assert resp.status_code == 200
    items = resp.json().get("items") or resp.json()
    assert len(items) >= 2


async def test_get_camera_by_id(
    app_client, db_session, tenant_a, client_admin_a, client_admin_a_token
):
    cam = await create_camera(db_session, tenant_a.client_id, "GetMe")
    await db_session.commit()

    resp = await app_client.get(
        f"/api/cameras/{cam.camera_id}", headers=auth_headers(client_admin_a_token)
    )
    assert resp.status_code == 200
    assert resp.json()["camera_id"] == str(cam.camera_id)


async def test_get_camera_nonexistent_returns_404(
    app_client, db_session, client_admin_a_token
):
    await db_session.commit()
    resp = await app_client.get(
        f"/api/cameras/{uuid.uuid4()}", headers=auth_headers(client_admin_a_token)
    )
    assert resp.status_code == 404


# ── Update / Delete ───────────────────────────────────────────────────────────

async def test_update_camera_name(
    app_client, db_session, tenant_a, client_admin_a, client_admin_a_token
):
    cam = await create_camera(db_session, tenant_a.client_id, "Old Name")
    await db_session.commit()

    resp = await app_client.put(
        f"/api/cameras/{cam.camera_id}",
        json={"name": "New Name"},
        headers=auth_headers(client_admin_a_token),
    )
    assert resp.status_code in (200, 204)

    get_resp = await app_client.get(
        f"/api/cameras/{cam.camera_id}", headers=auth_headers(client_admin_a_token)
    )
    assert get_resp.json()["name"] == "New Name"


async def test_delete_camera(
    app_client, db_session, tenant_a, client_admin_a, client_admin_a_token
):
    cam = await create_camera(db_session, tenant_a.client_id, "DeleteMe")
    await db_session.commit()

    resp = await app_client.delete(
        f"/api/cameras/{cam.camera_id}", headers=auth_headers(client_admin_a_token)
    )
    assert resp.status_code in (200, 204)

    get_resp = await app_client.get(
        f"/api/cameras/{cam.camera_id}", headers=auth_headers(client_admin_a_token)
    )
    assert get_resp.status_code == 404


# ── ONVIF test connection ─────────────────────────────────────────────────────

async def test_camera_test_connection_success(
    app_client, db_session, tenant_a, client_admin_a, client_admin_a_token
):
    cam = await create_camera(db_session, tenant_a.client_id, "TestConn")
    await db_session.commit()

    with patch(
        "app.api.cameras.ONVIFController",
        return_value=_mock_onvif(),
    ):
        resp = await app_client.post(
            f"/api/cameras/{cam.camera_id}/test",
            headers=auth_headers(client_admin_a_token),
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("status") in ("ok", "success", "ONLINE")


async def test_camera_test_connection_offline(
    app_client, db_session, tenant_a, client_admin_a, client_admin_a_token
):
    cam = await create_camera(db_session, tenant_a.client_id, "Offline")
    await db_session.commit()

    offline_ctrl = AsyncMock()
    offline_ctrl.get_rtsp_url = AsyncMock(
        side_effect=Exception("Connection refused")
    )

    with patch("app.api.cameras.ONVIFController", return_value=offline_ctrl):
        resp = await app_client.post(
            f"/api/cameras/{cam.camera_id}/test",
            headers=auth_headers(client_admin_a_token),
        )
    # Should return 200 with offline/error status, not 500
    assert resp.status_code in (200, 503)
    if resp.status_code == 200:
        assert resp.json().get("status") in ("offline", "error", "OFFLINE")


# ── ROI ───────────────────────────────────────────────────────────────────────

async def test_save_roi_rect(
    app_client, db_session, tenant_a, client_admin_a, client_admin_a_token
):
    cam = await create_camera(db_session, tenant_a.client_id, "ROI Cam")
    await db_session.commit()

    roi = {"x": 50, "y": 100, "width": 800, "height": 600}
    resp = await app_client.post(
        f"/api/cameras/{cam.camera_id}/roi",
        json={"roi_rect": roi},
        headers=auth_headers(client_admin_a_token),
    )
    assert resp.status_code in (200, 204)

    get_resp = await app_client.get(
        f"/api/cameras/{cam.camera_id}", headers=auth_headers(client_admin_a_token)
    )
    assert get_resp.json()["roi_rect"] == roi


async def test_save_roi_with_faculty_zone(
    app_client, db_session, tenant_a, client_admin_a, client_admin_a_token
):
    cam = await create_camera(db_session, tenant_a.client_id, "Faculty Cam")
    await db_session.commit()

    resp = await app_client.post(
        f"/api/cameras/{cam.camera_id}/roi",
        json={
            "roi_rect":     {"x": 0, "y": 0, "width": 1920, "height": 1080},
            "faculty_zone": {"x": 0, "y": 0, "width": 300, "height": 300},
        },
        headers=auth_headers(client_admin_a_token),
    )
    assert resp.status_code in (200, 204)


# ── PTZ control ───────────────────────────────────────────────────────────────

async def test_ptz_absolute_move(
    app_client, db_session, tenant_a, client_admin_a, client_admin_a_token
):
    cam = await create_camera(db_session, tenant_a.client_id, "PTZ Cam")
    await db_session.commit()

    mock_ctrl = _mock_onvif()
    with patch("app.api.cameras.ONVIFController", return_value=mock_ctrl):
        resp = await app_client.post(
            f"/api/cameras/{cam.camera_id}/ptz-move",
            json={"mode": "absolute", "pan": 0.5, "tilt": -0.2, "zoom": 0.8, "speed": 0.5},
            headers=auth_headers(client_admin_a_token),
        )
    assert resp.status_code == 200
    mock_ctrl.absolute_move.assert_called_once()


async def test_ptz_relative_move(
    app_client, db_session, tenant_a, client_admin_a, client_admin_a_token
):
    cam = await create_camera(db_session, tenant_a.client_id, "PTZ Rel")
    await db_session.commit()

    mock_ctrl = _mock_onvif()
    with patch("app.api.cameras.ONVIFController", return_value=mock_ctrl):
        resp = await app_client.post(
            f"/api/cameras/{cam.camera_id}/ptz-move",
            json={"mode": "relative", "pan": 0.1, "tilt": 0.0, "zoom": 0.0},
            headers=auth_headers(client_admin_a_token),
        )
    assert resp.status_code == 200
    mock_ctrl.relative_move.assert_called_once()


async def test_ptz_status(
    app_client, db_session, tenant_a, client_admin_a, client_admin_a_token
):
    cam = await create_camera(db_session, tenant_a.client_id, "PTZ Status")
    await db_session.commit()

    mock_ctrl = _mock_onvif()
    with patch("app.api.cameras.ONVIFController", return_value=mock_ctrl):
        resp = await app_client.get(
            f"/api/cameras/{cam.camera_id}/ptz-status",
            headers=auth_headers(client_admin_a_token),
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "pan" in data or "position" in data


async def test_ptz_snapshot(
    app_client, db_session, tenant_a, client_admin_a, client_admin_a_token
):
    cam = await create_camera(db_session, tenant_a.client_id, "Snapshot Cam")
    await db_session.commit()

    mock_ctrl = _mock_onvif()
    with patch("app.api.cameras.ONVIFController", return_value=mock_ctrl):
        resp = await app_client.get(
            f"/api/cameras/{cam.camera_id}/snapshot",
            headers=auth_headers(client_admin_a_token),
        )
    assert resp.status_code == 200
    data = resp.json()
    # Should return base64-encoded image
    assert "image" in data or "data" in data or "url" in data


# ── Camera mode-specific fields ───────────────────────────────────────────────

async def test_monitoring_camera_can_set_monitoring_hours(
    app_client, db_session, tenant_a, client_admin_a, client_admin_a_token
):
    db_session.add(ClientNodeAssignmentFactory(
        client_id=tenant_a.client_id, node_id="test-node-aaaaaaaa"
    ))
    await db_session.commit()

    resp = await app_client.post(
        "/api/cameras",
        json={
            "name":              "Monitor Cam",
            "rtsp_url":          "rtsp://10.1.1.6:554/stream",
            "onvif_host":        "10.1.1.6",
            "onvif_port":        80,
            "onvif_username":    "admin_onvif",
            "onvif_password":    "Test@2024",
            "mode":              "MONITORING",
            "node_id":           "test-node-aaaaaaaa",
            "monitoring_hours":  {"start": "08:00", "end": "20:00"},
            "restricted_zone":   True,
            "alert_on_unknown":  True,
        },
        headers=auth_headers(client_admin_a_token),
    )
    assert resp.status_code in (200, 201)
    data = resp.json()
    assert data.get("mode") == "MONITORING"


# ── Camera quota enforcement ──────────────────────────────────────────────────

async def test_camera_quota_is_enforced(
    app_client, db_session, tenant_a, client_admin_a, client_admin_a_token
):
    """Attempting to exceed max_cameras should return 409."""
    from app.models.clients import Client
    from sqlalchemy import select

    # Set max_cameras to current count so next creation fails
    result = await db_session.execute(
        select(Client).where(Client.client_id == tenant_a.client_id)
    )
    cli = result.scalar_one()
    cli.max_cameras = 0  # block all cameras
    db_session.add(ClientNodeAssignmentFactory(
        client_id=tenant_a.client_id, node_id="test-node-aaaaaaaa"
    ))
    await db_session.commit()

    resp = await app_client.post(
        "/api/cameras",
        json={
            "name":     "Over Quota",
            "rtsp_url": "rtsp://10.1.1.99:554/stream",
            "onvif_host": "10.1.1.99", "onvif_port": 80,
            "onvif_username": "admin_onvif", "onvif_password": "Test@2024",
            "mode": "ATTENDANCE", "node_id": "test-node-aaaaaaaa",
        },
        headers=auth_headers(client_admin_a_token),
    )
    assert resp.status_code in (409, 403, 422)
