from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROLE_LABELS = {
    "regulator": "监管机构",
    "consumer": "消费者",
    "producer": "生产经营者",
    "general": "通用咨询",
}


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def generate_demo_report(report: dict[str, Any]) -> str:
    summary = report.get("summary") or {}
    lines = [
        "# 食品安全法律法规问答系统演示报告",
        "",
        "## 演示结论",
        "",
        f"- 场景总数：{summary.get('total', 0)}",
        f"- 角色路由通过：{summary.get('role_passed', 0)}",
        f"- 带来源回答：{summary.get('with_sources', 0)}",
        f"- 带生成护栏 trace：{summary.get('with_generation_guard', 0)}",
        f"- 整体通过：{str(bool(summary.get('passed'))).lower()}",
        "",
        "## 三角色场景",
        "",
    ]
    for item in report.get("items", []):
        expected = ROLE_LABELS.get(item.get("expected_role"), item.get("expected_role", "unknown"))
        actual = ROLE_LABELS.get(item.get("actual_role"), item.get("actual_role", "unknown"))
        guard = item.get("generation_guard") or {}
        guard_status = "通过" if guard.get("accepted") else "回退/本地检索"
        violations = ", ".join(guard.get("violations") or []) or "none"
        lines.extend(
            [
                f"### {item.get('title', '未命名场景')}",
                "",
                f"- 问题：{item.get('question', '')}",
                f"- 期望角色：{expected}",
                f"- 实际角色：{actual}",
                f"- 路由正确：{str(bool(item.get('role_ok'))).lower()}",
                f"- 来源数量：{item.get('sources', 0)}",
                f"- 护栏状态：{guard_status}；violations={violations}",
                "",
                "答案预览：",
                "",
                item.get("answer_preview", "").strip(),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Markdown demo report from demo_scenarios.json.")
    parser.add_argument("--input", default="reports/demo_scenarios.json")
    parser.add_argument("--output", default="reports/demo_report.md")
    args = parser.parse_args()
    markdown = generate_demo_report(load_report(Path(args.input)))
    print(markdown)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
