from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile


def verify_evidence_package(package_path: Path) -> dict[str, Any]:
    try:
        with ZipFile(package_path) as archive:
            manifest = json.loads(archive.read("MANIFEST.json").decode("utf-8"))
            files = {}
            for name, expected in (manifest.get("files") or {}).items():
                try:
                    data = archive.read(name)
                except KeyError:
                    files[name] = {"status": "missing"}
                    continue
                actual_size = len(data)
                actual_hash = hashlib.sha256(data).hexdigest()
                if actual_size != expected.get("size_bytes"):
                    status = "size_mismatch"
                elif actual_hash != expected.get("sha256"):
                    status = "hash_mismatch"
                else:
                    status = "ok"
                files[name] = {
                    "status": status,
                    "size_bytes": actual_size,
                    "sha256": actual_hash,
                }
    except (BadZipFile, KeyError, json.JSONDecodeError) as exc:
        return {"passed": False, "error": str(exc), "files": {}}
    return {
        "passed": all(item["status"] == "ok" for item in files.values()),
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify evidence package MANIFEST.json hashes.")
    parser.add_argument("package", nargs="?", default="dist/food-law-qa-evidence.zip")
    args = parser.parse_args()
    result = verify_evidence_package(Path(args.package))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
