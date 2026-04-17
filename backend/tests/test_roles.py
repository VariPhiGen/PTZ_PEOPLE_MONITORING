"""
test_roles.py — Role-Based Access Control (RBAC).

Verifies:
  • SUPER_ADMIN   — full platform CRUD
  • CLIENT_ADMIN  — CRUD on own client's resources; forbidden on other clients
  • VIEWER        — read-only; all write endpoints return 403
  • 403 on missing permissions regardless of token validity
"""
from __future__ import annotations

import time
import uuid

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from tests.conftest import (
    _make_token,
    auth_headers,
    create_camera,
    create_client,
    create_person,
    create_user,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _sa_token(user_id: str, email: str, name: str) -> str:
    return _make_token(user_id, email, name, "SUPER_ADMIN")


def _ca_token(user_id: str, email: str, name: str, client_id: str, slug: str) -> str:
    return _make_token(user_id, email, name, "CLIENT_ADMIN", client_id, slug)


def _vw_token(user_id: str, email: str, name: str, client_id: str, slug: str) -> str:
    return _make_token(user_id, email, name, "VIEWER", client_id, slug)


# ─────────────────────────────────────────────────────────────────────────────
# SUPER_ADMIN capabilities
# ─────────────────────────────────────────────────────────────────────────────

class TestSuperAdmin:

    async def test_can_list_all_clients(self, app_client, db_session, super_admin, super_admin_token):
        client = await create_client(db_session, "Corp X")
        await db_session.commit()
        resp = await app_client.get("/api/admin/clients", headers=auth_headers(super_admin_token))
        assert resp.status_code == 200

    async def test_can_create_client(self, app_client, db_session, super_admin, super_admin_token):
        await db_session.commit()
        payload = {
            "name": "New Corp",
            "slug": "new-corp-" + str(uuid.uuid4())[:8],
            "contact_name":  "Jane Doe",
            "contact_email": "jane@newcorp.test",
            "max_cameras":   30,
            "max_persons":   5_000,
            "node_assignments": [{"node_id": "test-node-aaaaaaaa", "max_cameras_on_node": 10}],
            "initial_admin": {
                "name": "Corp Admin",
                "email": f"admin+{uuid.uuid4().hex[:6]}@newcorp.test",
                "temp_password": "TempPass@1234",
            },
        }
        resp = await app_client.post(
            "/api/admin/clients", json=payload, headers=auth_headers(super_admin_token)
        )
        assert resp.status_code in (200, 201)

    async def test_can_list_all_users(self, app_client, db_session, super_admin, super_admin_token):
        await create_user(db_session, "some@user.test")
        await db_session.commit()
        resp = await app_client.get("/api/admin/users", headers=auth_headers(super_admin_token))
        assert resp.status_code == 200

    async def test_can_suspend_any_user(
        self, app_client, db_session, tenant_a, super_admin, super_admin_token
    ):
        user = await create_user(db_session, "suspend.me@test.com", client_id=tenant_a.client_id)
        await db_session.commit()
        resp = await app_client.put(
            f"/api/admin/users/{user.user_id}/status",
            json={"status": "SUSPENDED"},
            headers=auth_headers(super_admin_token),
        )
        assert resp.status_code in (200, 204)

    async def test_can_read_platform_analytics(
        self, app_client, db_session, super_admin, super_admin_token
    ):
        await db_session.commit()
        resp = await app_client.get(
            "/api/admin/analytics/platform", headers=auth_headers(super_admin_token)
        )
        assert resp.status_code == 200

    async def test_can_change_client_limits(
        self, app_client, db_session, tenant_a, super_admin, super_admin_token
    ):
        await db_session.commit()
        resp = await app_client.put(
            f"/api/admin/clients/{tenant_a.client_id}",
            json={"max_cameras": 100},
            headers=auth_headers(super_admin_token),
        )
        assert resp.status_code in (200, 204)


# ─────────────────────────────────────────────────────────────────────────────
# CLIENT_ADMIN capabilities
# ─────────────────────────────────────────────────────────────────────────────

class TestClientAdmin:

    async def test_can_create_camera_on_own_client(
        self, app_client, db_session, tenant_a, client_admin_a, client_admin_a_token
    ):
        # Assign node so quota check passes
        from tests.factories import ClientNodeAssignmentFactory
        db_session.add(ClientNodeAssignmentFactory(
            client_id=tenant_a.client_id, node_id="test-node-aaaaaaaa"
        ))
        await db_session.commit()

        resp = await app_client.post(
            "/api/cameras",
            json={
                "name":       "Admin Camera",
                "room_name":  "Lab 101",
                "rtsp_url":   "rtsp://10.0.0.1:554/stream",
                "onvif_host": "10.0.0.1",
                "onvif_port": 80,
                "onvif_username": "admin_onvif",
                "onvif_password": "Test@2024",
                "mode":       "ATTENDANCE",
                "node_id":    "test-node-aaaaaaaa",
            },
            headers=auth_headers(client_admin_a_token),
        )
        assert resp.status_code in (200, 201)

    async def test_cannot_access_admin_clients_endpoint(
        self, app_client, db_session, client_admin_a_token
    ):
        await db_session.commit()
        resp = await app_client.get("/api/admin/clients", headers=auth_headers(client_admin_a_token))
        assert resp.status_code == 403

    async def test_cannot_create_client(self, app_client, db_session, client_admin_a_token):
        await db_session.commit()
        resp = await app_client.post(
            "/api/admin/clients",
            json={"name": "Hacker Corp", "slug": "hacker"},
            headers=auth_headers(client_admin_a_token),
        )
        assert resp.status_code == 403

    async def test_cannot_modify_other_client_cameras(
        self, app_client, db_session, tenant_a, client_admin_a_token
    ):
        other_client = await create_client(db_session, "Other Corp")
        cam = await create_camera(db_session, other_client.client_id, "Their Camera")
        await db_session.commit()

        resp = await app_client.put(
            f"/api/cameras/{cam.camera_id}",
            json={"name": "Hijacked"},
            headers=auth_headers(client_admin_a_token),
        )
        assert resp.status_code in (403, 404)

    async def test_can_enroll_person_on_own_client(
        self, app_client, db_session, tenant_a, client_admin_a, client_admin_a_token
    ):
        await db_session.commit()
        resp = await app_client.post(
            "/api/enrollment/enroll",
            json={
                "person":     {"name": "New Student", "role": "STUDENT", "external_id": "EXT-001"},
                "image_keys": ["fake-key-1", "fake-key-2", "fake-key-3",
                               "fake-key-4", "fake-key-5"],
            },
            headers=auth_headers(client_admin_a_token),
        )
        # Accept both success and 400 (missing MinIO keys) — the route is reachable
        assert resp.status_code in (200, 201, 400, 422)

    async def test_can_create_viewer(
        self, app_client, db_session, tenant_a, client_admin_a, client_admin_a_token
    ):
        await db_session.commit()
        resp = await app_client.post(
            "/api/users/viewers",
            json={
                "name":     "New Viewer",
                "email":    f"viewer.{uuid.uuid4().hex[:6]}@alpha.test",
                "password": "View@pass1234",
            },
            headers=auth_headers(client_admin_a_token),
        )
        assert resp.status_code in (200, 201)

    async def test_cannot_suspend_user_from_other_client(
        self, app_client, db_session, client_admin_a_token
    ):
        other = await create_client(db_session, "Rival Corp")
        their_user = await create_user(
            db_session, "rival@test.com", client_id=other.client_id
        )
        await db_session.commit()
        resp = await app_client.put(
            f"/api/admin/users/{their_user.user_id}/status",
            json={"status": "SUSPENDED"},
            headers=auth_headers(client_admin_a_token),
        )
        assert resp.status_code == 403


# ─────────────────────────────────────────────────────────────────────────────
# VIEWER capabilities (read-only)
# ─────────────────────────────────────────────────────────────────────────────

class TestViewer:

    async def test_can_list_cameras(
        self, app_client, db_session, tenant_a, viewer_a, viewer_a_token
    ):
        await create_camera(db_session, tenant_a.client_id)
        await db_session.commit()
        resp = await app_client.get("/api/cameras", headers=auth_headers(viewer_a_token))
        assert resp.status_code == 200

    async def test_can_list_persons(
        self, app_client, db_session, tenant_a, viewer_a, viewer_a_token
    ):
        await create_person(db_session, tenant_a.client_id)
        await db_session.commit()
        resp = await app_client.get("/api/enrollment/list", headers=auth_headers(viewer_a_token))
        assert resp.status_code == 200

    async def test_cannot_create_camera(
        self, app_client, db_session, viewer_a, viewer_a_token
    ):
        await db_session.commit()
        resp = await app_client.post(
            "/api/cameras",
            json={"name": "Viewer Cam", "rtsp_url": "rtsp://x", "mode": "ATTENDANCE"},
            headers=auth_headers(viewer_a_token),
        )
        assert resp.status_code == 403

    async def test_cannot_delete_camera(
        self, app_client, db_session, tenant_a, viewer_a, viewer_a_token
    ):
        cam = await create_camera(db_session, tenant_a.client_id)
        await db_session.commit()
        resp = await app_client.delete(
            f"/api/cameras/{cam.camera_id}", headers=auth_headers(viewer_a_token)
        )
        assert resp.status_code == 403

    async def test_cannot_enroll_person(
        self, app_client, db_session, viewer_a, viewer_a_token
    ):
        await db_session.commit()
        resp = await app_client.post(
            "/api/enrollment/enroll",
            json={"person": {"name": "X"}, "image_keys": ["k1"]},
            headers=auth_headers(viewer_a_token),
        )
        assert resp.status_code == 403

    async def test_cannot_start_session(
        self, app_client, db_session, tenant_a, viewer_a, viewer_a_token
    ):
        cam = await create_camera(db_session, tenant_a.client_id)
        await db_session.commit()
        resp = await app_client.post(
            "/api/sessions/start",
            json={"camera_id": str(cam.camera_id), "mode": "ATTENDANCE", "roster_ids": []},
            headers=auth_headers(viewer_a_token),
        )
        assert resp.status_code == 403

    async def test_cannot_override_attendance(
        self, app_client, db_session, tenant_a, viewer_a, viewer_a_token
    ):
        await db_session.commit()
        fake_record_id = str(uuid.uuid4())
        resp = await app_client.post(
            f"/api/attendance/records/{fake_record_id}/override",
            json={"status": "P", "reason": "Manual override"},
            headers=auth_headers(viewer_a_token),
        )
        assert resp.status_code == 403

    async def test_cannot_force_sync_held(
        self, app_client, db_session, viewer_a, viewer_a_token
    ):
        await db_session.commit()
        resp = await app_client.post(
            f"/api/attendance/held/{uuid.uuid4()}/force-sync",
            headers=auth_headers(viewer_a_token),
        )
        assert resp.status_code == 403

    async def test_cannot_access_admin_users(
        self, app_client, db_session, viewer_a, viewer_a_token
    ):
        await db_session.commit()
        resp = await app_client.get("/api/admin/users", headers=auth_headers(viewer_a_token))
        assert resp.status_code == 403

    async def test_can_view_attendance(
        self, app_client, db_session, tenant_a, viewer_a, viewer_a_token
    ):
        await db_session.commit()
        resp = await app_client.get(
            "/api/attendance/sessions", headers=auth_headers(viewer_a_token)
        )
        assert resp.status_code == 200

    async def test_can_search_persons_by_text(
        self, app_client, db_session, tenant_a, viewer_a, viewer_a_token
    ):
        await create_person(db_session, tenant_a.client_id, "Alice Viewer")
        await db_session.commit()
        resp = await app_client.get(
            "/api/search/person?q=Alice", headers=auth_headers(viewer_a_token)
        )
        assert resp.status_code == 200

    async def test_cannot_create_viewer(
        self, app_client, db_session, viewer_a, viewer_a_token
    ):
        """Viewers cannot manage other users."""
        await db_session.commit()
        resp = await app_client.post(
            "/api/users/viewers",
            json={"name": "Another", "email": "another@test.com", "password": "P@ss1234"},
            headers=auth_headers(viewer_a_token),
        )
        assert resp.status_code == 403


# ─────────────────────────────────────────────────────────────────────────────
# No token at all → 401
# ─────────────────────────────────────────────────────────────────────────────

class TestUnauthenticated:

    async def test_cameras_requires_auth(self, app_client):
        resp = await app_client.get("/api/cameras")
        assert resp.status_code == 401

    async def test_enrollment_requires_auth(self, app_client):
        resp = await app_client.get("/api/enrollment/list")
        assert resp.status_code == 401

    async def test_sessions_requires_auth(self, app_client):
        resp = await app_client.get("/api/sessions/active")
        assert resp.status_code == 401

    async def test_health_is_public(self, app_client):
        resp = await app_client.get("/api/health")
        assert resp.status_code == 200
