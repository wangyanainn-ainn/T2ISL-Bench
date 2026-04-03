"""
Validate Text2SceneLeakageBench merged JSON (cases + seeds alignment).

This repository ships the merged release under benchmark_cases/.
Split artifacts (subset / remaining / warning) are not included; see `provenance` in the JSON.

Optional: export a root-level copy for tools that expect benchmark_cases.json:

    python run_pipeline.py --export
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

DEFAULT_CASES = Path("benchmark_cases/text2sceneleakagebench_cases_merged_v1.json")
DEFAULT_SEEDS = Path("benchmark_cases/text2sceneleakagebench_seeds_merged_v1.json")


def _extract_seed_id(case_id: str) -> Optional[str]:
    """Prefix before _aligned_ or _conflict_ (e.g. fire_extinguisher_conflict_1_... -> fire_extinguisher)."""
    m = re.match(r"^(.+?)_(?:aligned|conflict)_", case_id)
    return m.group(1) if m else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate merged cases vs seeds; optionally export benchmark_cases.json."
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES, help="Merged cases JSON path")
    parser.add_argument("--seeds", type=Path, default=DEFAULT_SEEDS, help="Merged seeds JSON path")
    parser.add_argument(
        "--export",
        type=Path,
        nargs="?",
        const=Path("benchmark_cases.json"),
        metavar="OUT.json",
        help="Write merged cases document to this file (default: benchmark_cases.json)",
    )
    args = parser.parse_args()

    if not args.cases.exists():
        print(f"Missing cases file: {args.cases}", file=sys.stderr)
        sys.exit(1)

    raw: Dict[str, Any] = json.loads(args.cases.read_text(encoding="utf-8"))
    cases: List[Dict[str, Any]] = raw.get("cases")  # type: ignore[assignment]
    if not isinstance(cases, list):
        print(f"Invalid cases JSON (expected 'cases' list): {args.cases}", file=sys.stderr)
        sys.exit(1)

    n = len(cases)
    expected = raw.get("num_cases")
    if isinstance(expected, int) and expected != n:
        print(f"Warning: num_cases={expected} but len(cases)={n}", file=sys.stderr)

    for c in cases:
        if not isinstance(c, dict) or not c.get("case_id"):
            print("Invalid case entry (expected dict with case_id)", file=sys.stderr)
            sys.exit(1)

    seed_ids_in_cases: Set[str] = set()
    for c in cases:
        sid = _extract_seed_id(str(c["case_id"]))
        if sid:
            seed_ids_in_cases.add(sid)
        else:
            print(f"Warning: could not parse seed prefix from case_id={c['case_id']!r}", file=sys.stderr)

    if args.seeds.exists():
        seeds_doc = json.loads(args.seeds.read_text(encoding="utf-8"))
        slist = seeds_doc.get("seeds")
        if not isinstance(slist, list):
            print(f"Invalid seeds JSON (expected 'seeds' list): {args.seeds}", file=sys.stderr)
            sys.exit(1)
        official = {str(s["seed_id"]) for s in slist if isinstance(s, dict) and s.get("seed_id")}
        missing = official - seed_ids_in_cases
        extra = seed_ids_in_cases - official
        if missing:
            print(f"Warning: seeds not found as case_id prefix: {sorted(missing)}", file=sys.stderr)
        if extra:
            print(f"Warning: case prefixes not listed in seeds: {sorted(extra)}", file=sys.stderr)
        print(f"[ok] cases={n} seed_prefixes_in_cases={len(seed_ids_in_cases)} seeds={len(official)}")
    else:
        print(f"[ok] cases={n} seed_prefixes_in_cases={len(seed_ids_in_cases)} (no seeds file: {args.seeds})")

    if args.export is not None:
        args.export.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {args.export}")


if __name__ == "__main__":
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass
    main()
