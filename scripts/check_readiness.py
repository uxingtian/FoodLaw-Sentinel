from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import settings
from app.readiness import check_readiness


def main() -> None:
    parser = argparse.ArgumentParser(description="Check production dependency readiness.")
    parser.add_argument("--output", default="reports/readiness.json")
    parser.add_argument("--timeout", type=float, default=3.0)
    args = parser.parse_args()
    report = check_readiness(settings, timeout=args.timeout)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
