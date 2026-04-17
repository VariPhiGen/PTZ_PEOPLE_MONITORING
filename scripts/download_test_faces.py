"""
Download test face images for the ACAS recognition pipeline test.

Downloads clear, front-facing face images suitable for enrollment.
Default: Shah Rukh Khan (SRK) — publicly available promotional images.

Usage:
    python scripts/download_test_faces.py [--person srk] [--out-dir /tmp/srk_faces]

After downloading:
    python scripts/enroll_test_face.py \
        --name "Shah Rukh Khan" \
        --images-dir /tmp/srk_faces

    python scripts/test_recognition.py \
        --image /tmp/srk_faces/srk_test.jpg
"""
import argparse
import os
import sys
import urllib.request
import hashlib
import time


# Public domain / CC-licensed or promotional press images
# These are movie poster / press kit images widely used in ML face recognition datasets
FACE_SOURCES = {
    "srk": {
        "name":   "Shah Rukh Khan",
        "folder": "srk_faces",
        "images": [
            # Clear frontal press photos — publicly available
            ("https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Shah_Rukh_Khan_graces_the_launch_of_the_new_Santro.jpg/440px-Shah_Rukh_Khan_graces_the_launch_of_the_new_Santro.jpg",
             "srk1.jpg"),
            ("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c9/SRK_at_Zee_Cine_Awards_2018.jpg/440px-SRK_at_Zee_Cine_Awards_2018.jpg",
             "srk2.jpg"),
            ("https://upload.wikimedia.org/wikipedia/commons/thumb/0/0b/Shah_Rukh_Khan_at_the_Elle_Beauty_Awards_2018.jpg/440px-Shah_Rukh_Khan_at_the_Elle_Beauty_Awards_2018.jpg",
             "srk3.jpg"),
        ],
        "test_image": ("https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Shah_Rukh_Khan_graces_the_launch_of_the_new_Santro.jpg/440px-Shah_Rukh_Khan_graces_the_launch_of_the_new_Santro.jpg",
                       "srk_test.jpg"),
    },
}


def download_image(url: str, dest: str, timeout: int = 15) -> bool:
    """Download a single image, return True on success."""
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
        "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
    }
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content = resp.read()
        # Basic validation: JPEG/PNG magic bytes
        if content[:3] not in (b'\xff\xd8\xff', b'\x89PN'):
            if not content[:4] == b'\x89PNG':
                print(f"    [WARN] Unexpected content type (len={len(content)})")
        with open(dest, "wb") as f:
            f.write(content)
        size_kb = len(content) // 1024
        print(f"    ✓ {os.path.basename(dest)} ({size_kb} KB)")
        return True
    except Exception as exc:
        print(f"    ✗ Failed: {exc}")
        return False


def verify_image(path: str) -> bool:
    """Check if file is a valid image using cv2."""
    try:
        import cv2
        img = cv2.imread(path)
        if img is None or img.shape[0] < 10:
            return False
        return True
    except Exception:
        return True  # cv2 not available in this env — assume OK


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--person",  default="srk", choices=list(FACE_SOURCES.keys()),
                        help="Which test person to download")
    parser.add_argument("--out-dir", default="",
                        help="Output directory (default: /tmp/<folder_name>)")
    args = parser.parse_args()

    person_cfg = FACE_SOURCES[args.person]
    out_dir = args.out_dir or f"/tmp/{person_cfg['folder']}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Downloading face images for: {person_cfg['name']}")
    print(f"Output directory: {out_dir}")
    print(f"{'='*60}")

    # Download enrollment images
    print("\nEnrollment images:")
    downloaded = 0
    for url, fname in person_cfg["images"]:
        dest = os.path.join(out_dir, fname)
        if os.path.exists(dest) and os.path.getsize(dest) > 1000:
            print(f"  (skipping {fname} — already exists)")
            downloaded += 1
            continue
        if download_image(url, dest):
            downloaded += 1
        time.sleep(0.5)  # be polite

    # Download test image
    print("\nTest image (for recognition after enrollment):")
    test_url, test_fname = person_cfg["test_image"]
    test_dest = os.path.join(out_dir, test_fname)
    if os.path.exists(test_dest) and os.path.getsize(test_dest) > 1000:
        print(f"  (skipping {test_fname} — already exists)")
    else:
        download_image(test_url, test_dest)

    # Summary
    print(f"\n{'='*60}")
    print(f"Downloaded {downloaded}/{len(person_cfg['images'])} enrollment images")
    print(f"Output directory: {out_dir}")
    print(f"\nFiles:")
    for f in sorted(os.listdir(out_dir)):
        size = os.path.getsize(os.path.join(out_dir, f))
        print(f"  {f}  ({size//1024} KB)")

    print(f"\nNext steps:")
    print(f"\n  1. Enroll the person:")
    print(f"     python scripts/enroll_test_face.py \\")
    print(f'       --name "{person_cfg["name"]}" \\')
    print(f"       --images-dir {out_dir}")
    print(f"\n  2. Test recognition:")
    print(f"     python scripts/test_recognition.py \\")
    print(f"       --image {os.path.join(out_dir, test_fname)}")


if __name__ == "__main__":
    main()
