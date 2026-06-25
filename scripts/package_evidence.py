from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT_DIR = Path(__file__).resolve().parent.parent


EVIDENCE_FILES = [
    "demo_report.md",
    "resume_summary.md",
    "acceptance_audit.md",
    "doctor.json",
    "eval.json",
    "benchmark.json",
    "readiness.json",
    "claim_verification.json",
    "demo_scenarios.json",
    "acceptance_audit.json",
]

PROJECT_FILES = [
    ".env.production.example",
    "docs/production_runbook.md",
]


def build_evidence_package(*, reports_dir: Path, output_path: Path, root_dir: Path | None = None) -> Path:
    root_dir = root_dir or ROOT_DIR
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"files": {}}
    with ZipFile(output_path, "w", compression=ZIP_DEFLATED) as archive:
        for filename in EVIDENCE_FILES:
            path = reports_dir / filename
            if path.exists() and path.is_file():
                arcname = f"reports/{filename}"
                _write_with_manifest(archive, manifest, path, arcname)
        for filename in PROJECT_FILES:
            path = root_dir / filename
            if path.exists() and path.is_file():
                _write_with_manifest(archive, manifest, path, filename)
        archive.writestr("MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    return output_path


def _write_with_manifest(archive: ZipFile, manifest: dict, path: Path, arcname: str) -> None:
    archive.write(path, arcname=arcname)
    content = path.read_bytes()
    manifest["files"][arcname] = {
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Package verified evidence artifacts for resume/demo delivery.")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--output", default="dist/food-law-qa-evidence.zip")
    args = parser.parse_args()
    package_path = build_evidence_package(reports_dir=Path(args.reports_dir), output_path=Path(args.output))
    print(str(package_path))


if __name__ == "__main__":
    main()
