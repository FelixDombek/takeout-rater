"""
Tests for docs/tools/infer_takeout_sidecar_schema.py

Runs the script in-process against the fixture files in
tests/fixtures/takeout_sidecars/ and validates the output shape.

Can also be run as a standalone script to inspect a real Takeout folder:

  python tests/test_infer_takeout_sidecar_schema.py <TAKEOUT_DIR>

On Windows, when .py files are associated with an editor, use:

  python tests\\test_infer_takeout_sidecar_schema.py "H:\\Takeout"
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from types import ModuleType
from typing import Any

REPO_ROOT = Path(__file__).parent.parent
SCRIPT_PATH = REPO_ROOT / "docs" / "tools" / "infer_takeout_sidecar_schema.py"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "takeout_sidecars"


def _load_script() -> ModuleType:
    """Load the script as a module without executing __main__."""
    module_name = "infer_takeout_sidecar_schema"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so @dataclass can look up the module in sys.modules
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _run_against_fixtures() -> dict[str, Any]:
    """Execute main() against the fixture directory, capture stdout as JSON."""
    mod = _load_script()

    # Patch sys.argv so argparse picks up our fixture path
    original_argv = sys.argv
    sys.argv = ["infer_takeout_sidecar_schema.py", str(FIXTURES_DIR)]

    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            ret = mod.main()
    finally:
        sys.argv = original_argv

    assert ret == 0, f"Script exited with code {ret}"
    return json.loads(buf.getvalue())


# ── fixtures ──────────────────────────────────────────────────────────────────

FIXTURE_COUNT = len(list(FIXTURES_DIR.glob("*.supplemental-metadata.json")))


# ── tests ─────────────────────────────────────────────────────────────────────


def test_fixture_count_nonzero() -> None:
    """Sanity: we have at least one fixture file."""
    assert FIXTURE_COUNT >= 3


def test_input_ok_equals_fixture_count() -> None:
    result = _run_against_fixtures()
    assert result["input"]["ok"] == FIXTURE_COUNT


def test_input_failed_is_zero() -> None:
    result = _run_against_fixtures()
    assert result["input"]["failed"] == 0


def test_schema_is_object() -> None:
    result = _run_against_fixtures()
    schema = result["schema"]
    assert "object" in schema["kinds"]


def test_top_level_expected_properties_present() -> None:
    result = _run_against_fixtures()
    props = result["schema"]["properties"]
    for expected in ("title", "creationTime", "url"):
        assert expected in props, f"Expected top-level property {expected!r} missing from schema"


def test_creation_time_nested_properties() -> None:
    result = _run_against_fixtures()
    creation_time = result["schema"]["properties"]["creationTime"]
    assert "object" in creation_time["kinds"]
    ct_props = creation_time["properties"]
    assert "timestamp" in ct_props, "creationTime.timestamp missing"
    assert "formatted" in ct_props, "creationTime.formatted missing"


def test_geo_data_is_optional() -> None:
    """geoData is absent in at least one fixture, so it must be optional."""
    result = _run_against_fixtures()
    schema = result["schema"]
    # geoData should appear in properties (it's in at least one fixture)
    assert "geoData" in schema["properties"], "geoData not found in schema at all"
    # But it should be optional because not all fixtures have it
    assert "geoData" in schema["optional"], "geoData should be optional (not in all fixtures)"


def test_photo_taken_time_is_optional() -> None:
    """photoTakenTime is absent in at least one fixture, so it must be optional."""
    result = _run_against_fixtures()
    schema = result["schema"]
    assert "photoTakenTime" in schema["properties"], "photoTakenTime not found in schema at all"
    assert "photoTakenTime" in schema["optional"], (
        "photoTakenTime should be optional (not in all fixtures)"
    )


def test_title_is_required() -> None:
    """title is present in every fixture, so it should be required."""
    result = _run_against_fixtures()
    schema = result["schema"]
    assert "title" in schema["required"], "title should be required (present in all fixtures)"


def test_url_is_required() -> None:
    """url is present in every fixture, so it should be required."""
    result = _run_against_fixtures()
    schema = result["schema"]
    assert "url" in schema["required"], "url should be required (present in all fixtures)"


def test_creation_time_is_required() -> None:
    """creationTime is present in every fixture, so it should be required."""
    result = _run_against_fixtures()
    schema = result["schema"]
    assert "creationTime" in schema["required"], (
        "creationTime should be required (present in all fixtures)"
    )


# ── standalone entrypoint ─────────────────────────────────────────────────────


def _print_summary(result: dict[str, Any]) -> None:
    """Print a concise human-readable summary of the inference result."""
    inp = result["input"]
    schema = result["schema"]

    print(f"Takeout sidecar schema inference")
    print(f"  Root      : {inp['root']}")
    print(f"  Pattern   : {inp['pattern']}")
    print(f"  Processed : {inp['files_processed']} files  (ok={inp['ok']}, failed={inp['failed']})")
    print()

    required = schema.get("required", [])
    optional = schema.get("optional", [])
    props = schema.get("properties", {})

    print(f"Top-level properties ({len(props)} total):")
    for name in sorted(props.keys()):
        flag = "required" if name in required else "optional"
        kinds = ", ".join(props[name].get("kinds", []))
        print(f"  {name:<30} [{flag}]  kinds={kinds}")


def main() -> int:
    """Standalone entrypoint: infer schema from a real Takeout folder and print a summary."""
    ap = argparse.ArgumentParser(
        prog="test_infer_takeout_sidecar_schema.py",
        description=(
            "Infer a schema from Google Photos Takeout sidecar JSON files and print a summary.\n\n"
            "On Windows, when .py files are associated with an editor, invoke via:\n"
            "  python tests\\test_infer_takeout_sidecar_schema.py \"H:\\Takeout\""
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "takeout_dir",
        metavar="TAKEOUT_DIR",
        help="Root directory of the Takeout folder to scan.",
    )
    args = ap.parse_args()

    takeout_path = Path(args.takeout_dir)
    if not takeout_path.exists():
        print(f"error: path does not exist: {takeout_path}", file=sys.stderr)
        return 1
    if not takeout_path.is_dir():
        print(f"error: path is not a directory: {takeout_path}", file=sys.stderr)
        return 1

    mod = _load_script()

    original_argv = sys.argv
    sys.argv = ["infer_takeout_sidecar_schema.py", str(takeout_path)]

    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            ret = mod.main()
    finally:
        sys.argv = original_argv

    if ret != 0:
        print("error: inference failed (no sidecar files found?)", file=sys.stderr)
        return ret

    result = json.loads(buf.getvalue())
    _print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
