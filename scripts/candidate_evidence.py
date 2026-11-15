# Copyright 2026 Co-developed by KDA (Kernel Design Agents)
"""Record task-owned evidence and compare candidates without running their code."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1
STORE = "candidate-evidence"


def _identifier(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError(
            "candidate IDs must use 1-128 letters, digits, dots, underscores or hyphens"
        )
    return value


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(data: bytes) -> dict:
    def invalid(value):
        raise ValueError(f"non-finite JSON number: {value}")

    def floating(value):
        number = float(value)
        if not math.isfinite(number):
            invalid(value)
        return number

    value = json.loads(
        data, object_pairs_hook=_object, parse_constant=invalid, parse_float=floating
    )
    if not isinstance(value, dict):
        raise ValueError("report must be a JSON object")
    return value


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _relative(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("input paths must be nonempty relative file paths")
    path = Path(value)
    if not path.parts or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"path must stay inside the workspace: {value}")
    return path


def _workspace(path: Path) -> Path:
    root = Path(path).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("workspace must be an existing directory")
    return root


def _file(root: Path, value: str) -> Path:
    relative = _relative(value)
    path = root
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"input symlinks are not supported: {value}")
    if not path.resolve(strict=True).is_relative_to(root) or not path.is_file():
        raise ValueError(f"not a workspace file: {value}")
    return path


def _store(root: Path) -> Path:
    path = root / STORE
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        raise ValueError(f"{STORE} must be a directory, not a symlink or file")
    return path


def _validation(data: dict) -> dict:
    if data.get("status") not in ("passed", "failed"):
        raise ValueError("validation status must be 'passed' or 'failed'")
    return data


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _positive(value: object) -> float:
    if type(value) not in (int, float):
        raise ValueError("measurements must be positive finite numbers")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError("measurements must be positive finite numbers") from error
    if not math.isfinite(number) or number <= 0:
        raise ValueError("measurements must be positive finite numbers")
    return number


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    low, high = ordered[middle - 1], ordered[middle]
    return low + (high - low) / 2


def _benchmark(data: dict) -> dict:
    for field in ("environment", "protocol"):
        if not isinstance(data.get(field), dict) or not data[field]:
            raise ValueError(f"benchmark {field} must be a nonempty object")
    if not isinstance(data.get("measurements"), list) or not data["measurements"]:
        raise ValueError("benchmark measurements must be a nonempty list")
    seen = set()
    for item in data["measurements"]:
        if not isinstance(item, dict):
            raise ValueError("each measurement must be an object")
        for field in ("workload", "metric", "unit"):
            _text(item.get(field), field)
        key = (item["workload"], item["metric"])
        if key in seen:
            raise ValueError(f"duplicate workload/metric: {key}")
        seen.add(key)
        if item.get("direction") not in ("lower", "higher"):
            raise ValueError("measurement direction must be 'lower' or 'higher'")
        if ("value" in item) == ("samples" in item):
            raise ValueError("provide exactly one of value or samples per measurement")
        if "samples" in item:
            if not isinstance(item["samples"], list) or not item["samples"]:
                raise ValueError("samples must be a nonempty list")
            samples = [_positive(value) for value in item["samples"]]
            _positive(_median(samples))
        else:
            _positive(item["value"])
    return data


def _value(item: dict) -> float:
    if "samples" in item:
        return _median([_positive(value) for value in item["samples"]])
    return _positive(item["value"])


def _record(root: Path, candidate: str) -> tuple[dict, dict, dict | None]:
    directory = _store(root) / _identifier(candidate)
    if directory.is_symlink():
        raise ValueError("record directories must not be symlinks")
    manifest = _json(_file(root, f"{STORE}/{candidate}/record.json").read_bytes())
    if set(manifest) != {
        "schema_version",
        "candidate",
        "parent",
        "recorded_at",
        "sources",
        "reports",
    }:
        raise ValueError("record manifest has missing or unknown fields")
    _text(manifest["recorded_at"], "recorded_at")
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ValueError("unsupported record schema_version")
    if manifest.get("candidate") != candidate:
        raise ValueError("record candidate does not match its directory")
    if manifest.get("parent") is not None:
        if _identifier(manifest["parent"]) == candidate:
            raise ValueError("a candidate cannot be its own parent")
    sources = manifest.get("sources")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("record must contain source digests")
    for name, digest in sources.items():
        _relative(name)
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("invalid source digest")
    reports = manifest.get("reports")
    if not isinstance(reports, dict) or set(reports) not in (
        {"validation"},
        {"validation", "benchmark"},
    ):
        raise ValueError("record must identify validation and optional benchmark evidence")
    evidence = {}
    for name, expected in reports.items():
        data = _file(root, f"{STORE}/{candidate}/{name}.json").read_bytes()
        if _digest(data) != expected:
            raise ValueError(f"stored {name} evidence digest mismatch for {candidate}")
        evidence[name] = _json(data)
    validation = _validation(evidence["validation"])
    benchmark = _benchmark(evidence["benchmark"]) if "benchmark" in evidence else None
    if validation["status"] == "passed" and benchmark is None:
        raise ValueError("passing candidates require benchmark evidence")
    return manifest, validation, benchmark


def record_candidate(
    workspace: Path,
    candidate: str,
    sources: list[str],
    validation: str,
    benchmark: str | None = None,
    parent: str | None = None,
) -> Path:
    """Publish a new record; validation claims remain the report producer's responsibility."""
    root = _workspace(workspace)
    candidate = _identifier(candidate)
    if parent is not None:
        if _identifier(parent) == candidate:
            raise ValueError("a candidate cannot be its own parent")
        _record(root, parent)
    if not sources:
        raise ValueError("at least one source file is required")
    digests = {}
    for name in sources:
        path = _file(root, name)
        relative = path.relative_to(root).as_posix()
        if relative.split("/")[0] == STORE:
            raise ValueError("recorded evidence cannot itself be a candidate source")
        if relative in digests:
            raise ValueError(f"duplicate source file: {relative}")
        digests[relative] = _digest(path.read_bytes())
    raw = {"validation": _file(root, validation).read_bytes()}
    validation_data = _validation(_json(raw["validation"]))
    if benchmark is not None:
        raw["benchmark"] = _file(root, benchmark).read_bytes()
        _benchmark(_json(raw["benchmark"]))
    elif validation_data["status"] == "passed":
        raise ValueError("passing candidates require a benchmark report")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "candidate": candidate,
        "parent": parent,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "sources": digests,
        "reports": {name: _digest(data) for name, data in raw.items()},
    }
    store = _store(root)
    store.mkdir(exist_ok=True)
    destination = store / candidate
    try:
        destination.mkdir()
    except FileExistsError as error:
        raise ValueError(f"candidate record already exists: {candidate}") from error
    complete = False
    try:
        for name, data in raw.items():
            (destination / f"{name}.json").write_bytes(data)
        temporary = destination / ".record.json"
        temporary.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.rename(destination / "record.json")
        complete = True
    finally:
        if not complete:
            shutil.rmtree(destination)
    return destination / "record.json"


def compare_candidates(workspace: Path, baseline: str, candidates: list[str]) -> dict:
    """Compare only compatible, supplied correctness-passing evidence; never promote a candidate."""
    root = _workspace(workspace)
    if not candidates or len(set(candidates)) != len(candidates) or baseline in candidates:
        raise ValueError("provide distinct candidates, different from the baseline")
    baseline_manifest, baseline_validation, baseline_benchmark = _record(root, baseline)
    result = {
        "schema_version": SCHEMA_VERSION,
        "baseline": baseline,
        "baseline_sources": baseline_manifest["sources"],
        "baseline_environment": (baseline_benchmark or {}).get("environment"),
        "baseline_protocol": (baseline_benchmark or {}).get("protocol"),
        "candidates": [],
    }
    for candidate in sorted(candidates):
        manifest, validation, benchmark = _record(root, candidate)
        issues = []
        if baseline_validation["status"] != "passed":
            issues.append("baseline_validation_failed")
        if validation["status"] != "passed":
            issues.append("candidate_validation_failed")
        if baseline_benchmark is None:
            issues.append("missing_baseline_benchmark")
        if benchmark is None:
            issues.append("missing_candidate_benchmark")
        if baseline_benchmark is not None and benchmark is not None:
            for field in ("environment", "protocol"):
                if json.dumps(baseline_benchmark[field], sort_keys=True) != json.dumps(
                    benchmark[field], sort_keys=True
                ):
                    issues.append(f"{field}_mismatch")
        base = {
            (item["workload"], item["metric"]): item
            for item in (baseline_benchmark or {}).get("measurements", [])
        }
        other = {
            (item["workload"], item["metric"]): item
            for item in (benchmark or {}).get("measurements", [])
        }
        rows = []
        for workload, metric in sorted(base.keys() | other.keys()):
            key = (workload, metric)
            before, after = base.get(key), other.get(key)
            reasons = issues.copy()
            if before is None:
                reasons.append("missing_baseline_measurement")
            if after is None:
                reasons.append("missing_candidate_measurement")
            if before is not None and after is not None:
                for field in ("unit", "direction"):
                    if before[field] != after[field]:
                        reasons.append(f"{field}_mismatch")
            ratio = None
            if not reasons:
                numerator, denominator = _value(before), _value(after)
                if before["direction"] == "higher":
                    numerator, denominator = denominator, numerator
                ratio = numerator / denominator
                if not math.isfinite(ratio) or ratio <= 0:
                    reasons.append("unrepresentable_speedup")
                    ratio = None
            rows.append(
                {
                    "workload": workload,
                    "metric": metric,
                    "baseline_value": _value(before) if before is not None else None,
                    "candidate_value": _value(after) if after is not None else None,
                    "baseline_unit": before["unit"] if before is not None else None,
                    "candidate_unit": after["unit"] if after is not None else None,
                    "baseline_direction": before["direction"] if before is not None else None,
                    "candidate_direction": after["direction"] if after is not None else None,
                    "speedup": ratio,
                    "regression": ratio < 1 if ratio is not None else None,
                    "reasons": reasons,
                }
            )
        result["candidates"].append(
            {
                "candidate": candidate,
                "parent": manifest["parent"],
                "sources": manifest["sources"],
                "environment": (benchmark or {}).get("environment"),
                "protocol": (benchmark or {}).get("protocol"),
                "reported_validation": validation["status"],
                "baseline_reported_validation": baseline_validation["status"],
                "issues": issues,
                "comparison_complete": bool(rows) and all(not row["reasons"] for row in rows),
                "measurements": rows,
            }
        )
    return result


def _cell(value: object) -> str:
    return (
        html.escape(str(value), quote=False)
        .replace("|", "&#124;")
        .replace("\r", " ")
        .replace("\n", " ")
    )


def markdown_report(report: dict) -> str:
    lines = [
        f"# Candidate comparison against {_cell(report['baseline'])}",
        "",
        "Validation is reported evidence, not an independent correctness certification. "
        "Source digests describe recording time. No candidate is automatically promoted.",
        "",
        "| Candidate | Workload | Metric | Baseline | Candidate value | Speedup | Status |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for candidate in report["candidates"]:
        for row in candidate["measurements"] or [{"reasons": candidate["issues"]}]:
            values = []
            for field in ("baseline", "candidate"):
                value = row.get(f"{field}_value")
                values.append("-" if value is None else f"{value:.6g} {row[f'{field}_unit']}")
            ratio = row.get("speedup")
            status = ", ".join(row["reasons"]) or (
                "regression" if row["regression"] else "comparable"
            )
            cells = [
                candidate["candidate"],
                row.get("workload", "-"),
                row.get("metric", "-"),
                *values,
                "-" if ratio is None else f"{ratio:.3f}x",
                status,
            ]
            lines.append("| " + " | ".join(_cell(value) for value in cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    record = commands.add_parser("record", help="import existing validator/evaluator reports")
    record.add_argument("--workspace", required=True, type=Path)
    record.add_argument("--candidate", required=True)
    record.add_argument("--parent")
    record.add_argument("--source", required=True, action="append")
    record.add_argument("--validation", required=True)
    record.add_argument("--benchmark")
    compare = commands.add_parser("compare", help="compare stored evidence without running kernels")
    compare.add_argument("--workspace", required=True, type=Path)
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--candidate", required=True, action="append")
    compare.add_argument("--format", choices=("json", "markdown"), default="markdown")
    args = parser.parse_args(argv)
    try:
        if args.command == "record":
            print(
                record_candidate(
                    args.workspace,
                    args.candidate,
                    args.source,
                    args.validation,
                    args.benchmark,
                    args.parent,
                )
            )
        else:
            result = compare_candidates(args.workspace, args.baseline, args.candidate)
            print(
                json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
                if args.format == "json"
                else markdown_report(result),
                end="\n" if args.format == "json" else "",
            )
    except (OSError, ValueError, TypeError, OverflowError) as error:
        parser.exit(1, f"Evidence error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
