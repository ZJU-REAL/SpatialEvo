"""Run all public DGE example suites in one command."""

from __future__ import annotations

import scene_tasks_example
import single_image_tasks_example
import image_pair_tasks_example
import invalid_recovery_example


EXAMPLE_SUITES = [
    ("scene_tasks_example", scene_tasks_example.main),
    ("single_image_tasks_example", single_image_tasks_example.main),
    ("image_pair_tasks_example", image_pair_tasks_example.main),
    ("invalid_recovery_example", invalid_recovery_example.main),
]


def main() -> int:
    print("=" * 80)
    print("Run All DGE Examples")
    print("=" * 80)

    for name, entrypoint in EXAMPLE_SUITES:
        print(f"running: {name}")
        entrypoint()
        print(f"finished: {name}")
        print("-" * 80)

    print(f"summary: finished {len(EXAMPLE_SUITES)} example suites")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
