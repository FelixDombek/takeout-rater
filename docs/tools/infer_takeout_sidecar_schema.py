#!/usr/bin/env python3
"""
infer_takeout_sidecar_schema.py

Walk a Google Photos Takeout folder, read each *.supplemental-metadata.json file,
and infer an aggregate schema that includes all optional fields encountered.

Output: a single JSON document describing the inferred schema.

Notes
- This is a best-effort, data-driven schema inference (not a formal JSON Schema draft).
- It tries to be robust to type variations across files (e.g., int vs str vs null).
- It is incremental: it merges new observations into the schema as it scans.

Usage:
  python infer_takeout_sidecar_schema.py "D:/Photos/Takeout" --out schema.json
  python infer_takeout_sidecar_schema.py "D:/Photos" --pattern "*.supplemental-metadata.json"

Tips:
- On very large takeouts, you can add --limit to test quickly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import Any


def _python_type_name(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int) and not isinstance(v, bool):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "str"
    if isinstance(v, dict):
        return "object"
    if isinstance(v, list):
        return "array"
    return f"unknown:{type(v).__name__}"


@dataclass
class SchemaNode:
    """
    Represents an inferred schema node.
    kind: one of {"object","array","scalar","union"} (union used when multiple kinds exist)
    """

    kinds: set[str] = field(default_factory=set)
    scalar_types: set[str] = field(default_factory=set)
    properties: dict[str, "SchemaNode"] = field(default_factory=dict)
    required_count: dict[str, int] = field(default_factory=dict)
    object_instances: int = 0
    item_schema: "SchemaNode | None" = None
    array_instances: int = 0
    array_lengths: dict[str, int] = field(default_factory=dict)
    seen: int = 0
    examples: list[str] = field(default_factory=list)

    def merge_value(self, value: Any, *, example_budget: int = 3) -> None:
        self.seen += 1
        t = _python_type_name(value)
        if t == "object":
            self.kinds.add("object")
            self.object_instances += 1
            assert isinstance(value, dict)
            for k in value.keys():
                self.required_count[k] = self.required_count.get(k, 0) + 1
            for k, v in value.items():
                if k not in self.properties:
                    self.properties[k] = SchemaNode()
                self.properties[k].merge_value(v, example_budget=example_budget)
        elif t == "array":
            self.kinds.add("array")
            self.array_instances += 1
            assert isinstance(value, list)
            if "min" not in self.array_lengths:
                self.array_lengths["min"] = len(value)
                self.array_lengths["max"] = len(value)
            else:
                self.array_lengths["min"] = min(int(self.array_lengths["min"]), len(value))
                self.array_lengths["max"] = max(int(self.array_lengths["max"]), len(value))
            if self.item_schema is None:
                self.item_schema = SchemaNode()
            for item in value:
                self.item_schema.merge_value(item, example_budget=example_budget)
        else:
            self.kinds.add("scalar")
            self.scalar_types.add(t)
            if len(self.examples) < example_budget:
                try:
                    s = json.dumps(value, ensure_ascii=False)
                except Exception:
                    s = repr(value)
                if len(s) > 200:
                    s = s[:197] + "..."
                self.examples.append(s)

    def to_dict(self) -> dict[str, Any]:
        kinds_sorted = sorted(self.kinds) if self.kinds else []

        node: dict[str, Any] = {
            "kinds": kinds_sorted,
            "seen": self.seen,
        }

        if "scalar" in self.kinds:
            node["scalar_types"] = sorted(self.scalar_types)
            if self.examples:
                node["examples"] = self.examples

        if "object" in self.kinds:
            props: dict[str, Any] = {}
            for k in sorted(self.properties.keys()):
                props[k] = self.properties[k].to_dict()

            required: list[str] = []
            optional: list[str] = []
            for k in sorted(self.properties.keys()):
                present = self.required_count.get(k, 0)
                if self.object_instances > 0 and present == self.object_instances:
                    required.append(k)
                else:
                    optional.append(k)

            node["object_instances"] = self.object_instances
            node["required"] = required
            node["optional"] = optional
            node["properties"] = props

        if "array" in self.kinds:
            node["array_instances"] = self.array_instances
            if self.array_lengths:
                node["array_length"] = {
                    "min": int(self.array_lengths["min"]),
                    "max": int(self.array_lengths["max"]),
                }
            node["items"] = (
                self.item_schema.to_dict() if self.item_schema else {"kinds": [], "seen": 0}
            )

        return node


def iter_files(root: str, pattern: str) -> list[str]:
    out: list[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fnmatch(fn, pattern):
                out.append(os.path.join(dirpath, fn))
    return out


def load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Infer aggregate schema from Google Photos Takeout sidecar JSON files."
    )
    ap.add_argument(
        "root",
        help="Root directory to scan (e.g., the directory containing Takeout/ or Takeout/ itself).",
    )
    ap.add_argument(
        "--pattern",
        default="*.supplemental-metadata.json",
        help="Glob pattern for sidecar files.",
    )
    ap.add_argument(
        "--out",
        default="",
        help="Output path for schema JSON. If omitted, prints to stdout.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max number of JSON files to process (0 = no limit).",
    )
    ap.add_argument(
        "--show-progress",
        action="store_true",
        help="Print progress to stderr.",
    )
    args = ap.parse_args()

    files = iter_files(args.root, args.pattern)
    files.sort()

    if not files:
        print(
            f"No files matched pattern {args.pattern!r} under {args.root!r}",
            file=sys.stderr,
        )
        return 2

    if args.limit and args.limit > 0:
        files = files[: args.limit]

    schema = SchemaNode()
    ok = 0
    failed = 0

    for i, path in enumerate(files, start=1):
        try:
            data = load_json(path)
            schema.merge_value(data)
            ok += 1
        except Exception as e:
            failed += 1
            if args.show_progress:
                print(f"[{i}/{len(files)}] FAIL {path}: {e}", file=sys.stderr)

        if args.show_progress and (i % 500 == 0 or i == len(files)):
            print(f"[{i}/{len(files)}] processed (ok={ok}, failed={failed})", file=sys.stderr)

    result = {
        "input": {
            "root": os.path.abspath(args.root),
            "pattern": args.pattern,
            "files_processed": len(files),
            "ok": ok,
            "failed": failed,
        },
        "schema": schema.to_dict(),
    }

    out_json = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        out_dir = os.path.dirname(os.path.abspath(args.out))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(out_json)
            f.write("\n")
    else:
        print(out_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
