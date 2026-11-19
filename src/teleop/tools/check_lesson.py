"""Enforce the Lesson Rule (see AGENTS.md): all code maps to the lesson HTML.

Every tracked source file must be *mentioned by its repo-relative path*
somewhere in ``lesson/index.html``. A mention means there is a section
walking through that file — this check keeps code and lesson from drifting
apart. It runs as ``uv run check-lesson`` and exits non-zero on violations.
"""

from __future__ import annotations

import sys

from teleop import REPO_ROOT

LESSON = REPO_ROOT / "lesson" / "index.html"

# Files that must each have a walkthrough in the lesson.
TRACKED_GLOBS = [
    "src/teleop/**/*.py",
    "src/teleop/sources/webxr/index.html",
    "assets/*.xml",
    "pyproject.toml",
]
# Plumbing with no teaching content of its own.
EXEMPT = {
    "src/teleop/__init__.py",
    "src/teleop/sources/__init__.py",
    "src/teleop/tools/__init__.py",
}


def tracked_files() -> list[str]:
    files: set[str] = set()
    for pattern in TRACKED_GLOBS:
        for path in REPO_ROOT.glob(pattern):
            if path.is_file() and "__pycache__" not in path.parts:
                files.add(path.relative_to(REPO_ROOT).as_posix())
    return sorted(files - EXEMPT)


def main() -> None:
    if not LESSON.exists():
        sys.exit(f"check-lesson: FAIL — lesson file not found: {LESSON}")
    html = LESSON.read_text(encoding="utf-8")

    missing = [f for f in tracked_files() if f not in html]
    for f in tracked_files():
        mark = "MISS" if f in missing else "ok"
        print(f"  [{mark:>4}] {f}")

    if missing:
        sys.exit(
            f"check-lesson: FAIL — {len(missing)} file(s) have no walkthrough in "
            f"lesson/index.html. The Lesson Rule (AGENTS.md): every code change "
            f"updates the lesson in the same change."
        )
    print(f"check-lesson: OK — all {len(tracked_files())} tracked files appear in the lesson.")


if __name__ == "__main__":
    main()
