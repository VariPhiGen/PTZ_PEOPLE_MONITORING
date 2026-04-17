"""
Enroll a test person from local images via the ACAS API.

Usage:
    python scripts/enroll_test_face.py \
        --name "Shah Rukh Khan" \
        --images-dir /tmp/srk_faces \
        [--api http://localhost:18000] \
        [--token <jwt>]

If --token is omitted, the script logs in with default admin credentials
(admin@acas.local / Admin@2024!) to get a JWT.

The script:
  1. Logs in → gets JWT
  2. Uploads each image via POST /api/enrollment/upload (quality gate)
  3. Enrolls the person via POST /api/enrollment/enroll
  4. Prints the person_id for use in test_recognition.py
"""
import argparse
import json
import os
import sys
import urllib.request
import urllib.parse
import urllib.error


def api_call(url: str, method: str = "GET", data=None, files=None, token: str = "") -> dict:
    """Minimal HTTP client using only stdlib."""
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    if files:
        # Multipart form — build manually
        boundary = "----ACAS_BOUNDARY_7f3k9x"
        parts = []
        for name, (filename, content, content_type) in files.items():
            parts.append(
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                f"Content-Type: {content_type}\r\n\r\n"
            )
        # Rebuild as bytes
        body = b""
        for name, (filename, content, content_type) in files.items():
            body += (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                f"Content-Type: {content_type}\r\n\r\n"
            ).encode()
            body += content if isinstance(content, bytes) else content.encode()
            body += b"\r\n"
        if data:
            for k, v in data.items():
                body += (
                    f"--{boundary}\r\n"
                    f'Content-Disposition: form-data; name="{k}"\r\n\r\n'
                    f"{v}\r\n"
                ).encode()
        body += f"--{boundary}--\r\n".encode()
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
    elif data is not None:
        body = json.dumps(data).encode()
        headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
    else:
        req = urllib.request.Request(url, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"  HTTP {e.code}: {body[:300]}")
        raise


def login(api: str, email: str, password: str) -> str:
    """Login and return JWT token."""
    result = api_call(
        f"{api}/api/auth/login",
        method="POST",
        data={"email": email, "password": password},
    )
    token = result.get("access_token") or result.get("token")
    if not token:
        raise RuntimeError(f"Login failed: {result}")
    return token


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name",       required=True, help="Person's full name")
    parser.add_argument("--images-dir", required=True, help="Directory with face images (.jpg/.png)")
    parser.add_argument("--api",        default="http://localhost:18000")
    parser.add_argument("--token",      default="", help="JWT token (login auto if omitted)")
    parser.add_argument("--email",      default="admin@acas.local")
    parser.add_argument("--password",   default="Admin@2024!")
    parser.add_argument("--role",       default="STUDENT")
    parser.add_argument("--dataset-id", default="", help="Face dataset UUID (auto if omitted)")
    args = parser.parse_args()

    api = args.api.rstrip("/")

    # ── Step 1: Authenticate ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("STEP 1 — Authenticating")
    token = args.token
    if not token:
        print(f"  Logging in as {args.email}...")
        token = login(api, args.email, args.password)
        print(f"  Token obtained: {token[:20]}...")
    else:
        print(f"  Using provided token.")

    # ── Step 2: Collect images ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("STEP 2 — Collecting images")
    img_dir = args.images_dir
    if not os.path.isdir(img_dir):
        print(f"  [ERROR] Directory not found: {img_dir}")
        sys.exit(1)

    images = sorted([
        os.path.join(img_dir, f) for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    if not images:
        print(f"  [ERROR] No .jpg/.png files found in {img_dir}")
        sys.exit(1)
    print(f"  Found {len(images)} image(s)")
    for img in images:
        print(f"    {os.path.basename(img)}")

    # ── Step 3: Upload images (quality gate) ─────────────────────────────────
    print(f"\n{'='*60}")
    print("STEP 3 — Uploading images (quality gate)")
    image_keys: list[str] = []
    failed = 0

    for img_path in images:
        with open(img_path, "rb") as f:
            img_bytes = f.read()
        fname = os.path.basename(img_path)
        ct = "image/jpeg" if img_path.lower().endswith(".jpg") or img_path.lower().endswith(".jpeg") else "image/png"

        try:
            result = api_call(
                f"{api}/api/enrollment/upload",
                method="POST",
                files={"file": (fname, img_bytes, ct)},
                token=token,
            )
            if result.get("passed"):
                quality = result.get("quality", {})
                print(f"  ✓ {fname}  IOD={quality.get('iod_px','?')}px  sharpness={quality.get('sharpness','?')}  key_len={len(result.get('temp_key',''))}")
                image_keys.append(result["temp_key"])
            else:
                print(f"  ✗ {fname}  REJECTED: {result.get('reason')}")
                failed += 1
        except Exception as exc:
            print(f"  ✗ {fname}  ERROR: {exc}")
            failed += 1

    if not image_keys:
        print(f"\n  [FAIL] No images passed quality gate. Check faces are clear and front-facing.")
        sys.exit(1)
    print(f"\n  {len(image_keys)} image(s) passed, {failed} rejected")

    # ── Step 4: Enroll ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("STEP 4 — Enrolling person")
    enroll_payload: dict = {
        "name":       args.name,
        "role":       args.role,
        "image_keys": image_keys,
    }
    if args.dataset_id:
        enroll_payload["dataset_id"] = args.dataset_id

    result = api_call(
        f"{api}/api/enrollment/enroll",
        method="POST",
        data=enroll_payload,
        token=token,
    )
    person_id = result.get("person_id") or result.get("id")
    print(f"  Enrolled: {args.name}")
    print(f"  person_id: {person_id}")
    print(f"  Embeddings: {result.get('embedding_count', '?')}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("DONE")
    print(f"  Name:      {args.name}")
    print(f"  Person ID: {person_id}")
    print(f"  Images:    {len(image_keys)} enrolled")
    print(f"\nTo test recognition:")
    print(f"  python scripts/test_recognition.py --image <path_to_test_image.jpg> --expected-id {person_id}")

    # Save person_id for convenience
    with open("/tmp/last_enrolled_person.txt", "w") as f:
        f.write(f"{person_id}\n{args.name}\n")
    print(f"\n  person_id saved to /tmp/last_enrolled_person.txt")


if __name__ == "__main__":
    main()
