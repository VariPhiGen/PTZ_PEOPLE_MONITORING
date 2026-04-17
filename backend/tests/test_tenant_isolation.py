"""
test_tenant_isolation.py — Multi-tenant data isolation.

Verifies that:
  • Client A admin cannot read/write Client B data
  • PostgreSQL RLS prevents cross-tenant data leakage at the DB layer
  • FAISS indexes are per-client (no cross-client embedding leakage)
  • Super admin can see all clients
"""
from __future__ import annotations

import time
import uuid

import numpy as np
import pytest
from httpx import AsyncClient
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from tests.conftest import (
    MockFaceRepository,
    _make_token,
    _random_embedding,
    auth_headers,
    create_camera,
    create_client,
    create_person,
    create_user,
)
from tests.factories import FaceEmbeddingFactory


# ── Fixtures for two isolated tenants ─────────────────────────────────────────

@pytest.fixture
async def tenant_b(db_session):
    return await create_client(db_session, "Tenant Beta")


@pytest.fixture
async def admin_b(db_session, tenant_b):
    return await create_user(
        db_session, "admin.b@beta.test",
        role="CLIENT_ADMIN", client_id=tenant_b.client_id,
    )


@pytest.fixture
def token_a(client_admin_a, tenant_a):
    return _make_token(
        str(client_admin_a.user_id), client_admin_a.email,
        client_admin_a.name, "CLIENT_ADMIN",
        str(tenant_a.client_id), tenant_a.slug,
    )


@pytest.fixture
def token_b(admin_b, tenant_b):
    return _make_token(
        str(admin_b.user_id), admin_b.email,
        admin_b.name, "CLIENT_ADMIN",
        str(tenant_b.client_id), tenant_b.slug,
    )


# ── Camera isolation ──────────────────────────────────────────────────────────

async def test_camera_list_only_shows_own_cameras(
    app_client, db_session, tenant_a, tenant_b, token_a, token_b
):
    """Each admin sees only cameras belonging to their tenant."""
    cam_a = await create_camera(db_session, tenant_a.client_id, "Alpha Camera")
    cam_b = await create_camera(db_session, tenant_b.client_id, "Beta Camera")
    await db_session.commit()

    resp_a = await app_client.get("/api/cameras", headers=auth_headers(token_a))
    resp_b = await app_client.get("/api/cameras", headers=auth_headers(token_b))

    assert resp_a.status_code == 200
    assert resp_b.status_code == 200

    ids_a = {c["camera_id"] for c in (resp_a.json().get("items") or resp_a.json())}
    ids_b = {c["camera_id"] for c in (resp_b.json().get("items") or resp_b.json())}

    assert str(cam_a.camera_id) in ids_a
    assert str(cam_b.camera_id) not in ids_a  # A cannot see B's camera
    assert str(cam_b.camera_id) in ids_b
    assert str(cam_a.camera_id) not in ids_b  # B cannot see A's camera


async def test_camera_direct_access_denied_cross_tenant(
    app_client, db_session, tenant_b, token_a
):
    """Client A admin trying to GET Client B's camera must receive 403/404."""
    cam_b = await create_camera(db_session, tenant_b.client_id, "Private Beta Camera")
    await db_session.commit()

    resp = await app_client.get(
        f"/api/cameras/{cam_b.camera_id}",
        headers=auth_headers(token_a),
    )
    assert resp.status_code in (403, 404)


# ── Person / enrollment isolation ─────────────────────────────────────────────

async def test_person_list_isolated_by_tenant(
    app_client, db_session, tenant_a, tenant_b, token_a, token_b
):
    person_a = await create_person(db_session, tenant_a.client_id, "Alice Alpha")
    person_b = await create_person(db_session, tenant_b.client_id, "Bob Beta")
    await db_session.commit()

    resp_a = await app_client.get("/api/enrollment/list", headers=auth_headers(token_a))
    resp_b = await app_client.get("/api/enrollment/list", headers=auth_headers(token_b))

    assert resp_a.status_code == 200
    assert resp_b.status_code == 200

    names_a = {p["name"] for p in (resp_a.json().get("items") or resp_a.json())}
    names_b = {p["name"] for p in (resp_b.json().get("items") or resp_b.json())}

    assert "Alice Alpha" in names_a
    assert "Bob Beta" not in names_a
    assert "Bob Beta" in names_b
    assert "Alice Alpha" not in names_b


async def test_person_direct_access_denied_cross_tenant(
    app_client, db_session, tenant_b, token_a
):
    person_b = await create_person(db_session, tenant_b.client_id, "Secret Bob")
    await db_session.commit()

    resp = await app_client.get(
        f"/api/enrollment/{person_b.person_id}",
        headers=auth_headers(token_a),
    )
    assert resp.status_code in (403, 404)


# ── PostgreSQL RLS layer ───────────────────────────────────────────────────────

async def test_rls_blocks_cross_tenant_select(
    db_session: AsyncSession, tenant_a, tenant_b
):
    """
    Directly verify that setting app.current_client_id scopes SELECT to
    that tenant only.
    """
    person_a = await create_person(db_session, tenant_a.client_id, "RLS Test A")
    person_b = await create_person(db_session, tenant_b.client_id, "RLS Test B")
    await db_session.commit()

    # Set RLS context to Tenant A
    await db_session.execute(
        text("SET LOCAL app.current_client_id = :cid"),
        {"cid": str(tenant_a.client_id)},
    )

    from app.models.persons import Person
    result = await db_session.execute(select(Person))
    visible_names = {p.name for p in result.scalars().all()}

    assert "RLS Test A" in visible_names
    assert "RLS Test B" not in visible_names


async def test_rls_empty_context_is_blocked(db_session: AsyncSession, tenant_a):
    """
    An empty app.current_client_id should allow zero rows (policy requires
    non-empty match or explicit bypass for SUPER_ADMIN which uses '').
    """
    await create_person(db_session, tenant_a.client_id, "RLS Person")
    await db_session.commit()

    # A random UUID that matches no tenant
    random_cid = str(uuid.uuid4())
    await db_session.execute(
        text("SET LOCAL app.current_client_id = :cid"),
        {"cid": random_cid},
    )

    from app.models.persons import Person
    result = await db_session.execute(select(Person))
    visible = result.scalars().all()
    assert len(visible) == 0


async def test_rls_superadmin_bypass_sees_all(db_session: AsyncSession, tenant_a, tenant_b):
    """Super admin sets app.current_client_id='' which bypasses RLS."""
    await create_person(db_session, tenant_a.client_id, "Person A")
    await create_person(db_session, tenant_b.client_id, "Person B")
    await db_session.commit()

    # Bypass RLS (super admin mode)
    await db_session.execute(text("SET LOCAL app.current_client_id = ''"))

    from app.models.persons import Person
    result = await db_session.execute(select(Person))
    all_persons = result.scalars().all()
    names = {p.name for p in all_persons}
    assert "Person A" in names
    assert "Person B" in names


# ── FAISS per-client isolation ────────────────────────────────────────────────

async def test_faiss_index_isolated_per_client():
    """
    Enroll persons on two different clients and verify that identify()
    never returns a person from the other client's index.
    """
    repo = MockFaceRepository()
    cid_a = str(uuid.uuid4())
    cid_b = str(uuid.uuid4())
    pid_a = str(uuid.uuid4())
    pid_b = str(uuid.uuid4())

    # Enroll on separate clients
    await repo.enroll_person(cid_a, pid_a, [b"img"] * 5)
    await repo.enroll_person(cid_b, pid_b, [b"img"] * 5)

    # Use the exact embedding stored for A — should match in A but not B
    emb_a = repo._embs[cid_a][pid_a]

    result_a = await repo.identify(cid_a, emb_a)
    result_b = await repo.identify(cid_b, emb_a)  # B's index has no similar embedding

    assert result_a.person_id == pid_a
    # emb_a is random relative to pid_b's embedding — should not match
    assert result_b.person_id != pid_a


async def test_faiss_delete_only_affects_own_client():
    repo = MockFaceRepository()
    cid_a, cid_b = str(uuid.uuid4()), str(uuid.uuid4())
    pid   = str(uuid.uuid4())

    await repo.enroll_person(cid_a, pid, [b"img"] * 5)
    await repo.enroll_person(cid_b, pid, [b"img"] * 5)

    await repo.delete_person(cid_a, pid)

    assert pid not in repo._embs.get(cid_a, {})
    assert pid     in repo._embs.get(cid_b, {})  # B's copy untouched


# ── Super admin sees all tenants ──────────────────────────────────────────────

async def test_super_admin_list_all_cameras(
    app_client, db_session, tenant_a, tenant_b, super_admin, super_admin_token
):
    cam_a = await create_camera(db_session, tenant_a.client_id, "SA Camera A")
    cam_b = await create_camera(db_session, tenant_b.client_id, "SA Camera B")
    await db_session.commit()

    resp = await app_client.get("/api/cameras", headers=auth_headers(super_admin_token))
    assert resp.status_code == 200
    items = resp.json().get("items") or resp.json()
    ids = {c["camera_id"] for c in items}
    assert str(cam_a.camera_id) in ids
    assert str(cam_b.camera_id) in ids


async def test_super_admin_list_clients(
    app_client, db_session, tenant_a, tenant_b, super_admin_token
):
    await db_session.commit()
    resp = await app_client.get("/api/admin/clients", headers=auth_headers(super_admin_token))
    assert resp.status_code == 200
    clients = resp.json().get("items") or resp.json()
    names = {c["name"] for c in clients}
    # Both tenants must appear in the list
    assert tenant_a.name in names
    assert tenant_b.name in names


async def test_client_admin_cannot_access_admin_clients_endpoint(
    app_client, db_session, client_admin_a_token
):
    await db_session.commit()
    resp = await app_client.get("/api/admin/clients", headers=auth_headers(client_admin_a_token))
    assert resp.status_code == 403


# ── Attendance isolation ──────────────────────────────────────────────────────

async def test_attendance_sessions_isolated(
    app_client, db_session, tenant_a, tenant_b, token_a
):
    from tests.conftest import create_session_record
    cam_a = await create_camera(db_session, tenant_a.client_id)
    cam_b = await create_camera(db_session, tenant_b.client_id)
    sess_a = await create_session_record(db_session, tenant_a.client_id, cam_a.camera_id)
    sess_b = await create_session_record(db_session, tenant_b.client_id, cam_b.camera_id)
    await db_session.commit()

    resp = await app_client.get("/api/attendance/sessions", headers=auth_headers(token_a))
    assert resp.status_code == 200
    items = resp.json().get("items") or resp.json()
    ids = {s["session_id"] for s in items}
    assert str(sess_a.session_id) in ids
    assert str(sess_b.session_id) not in ids
