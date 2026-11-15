# Copyright 2026 Co-developed by KDA (Kernel Design Agents)

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from scripts import candidate_evidence as evidence

SCRIPT = Path(evidence.__file__).resolve()


@pytest.fixture
def task(tmp_path):
    (tmp_path / "solution").mkdir()
    (tmp_path / "solution/kernel.py").write_text('LABEL = "example"\n', encoding="utf-8")
    (tmp_path / "validation.json").write_text('{"status": "passed"}\n')
    return tmp_path


def benchmark(value=2, **changes):
    result = {
        "environment": {"device": "test-cpu", "toolchain": "test-v1"},
        "protocol": {"method": "test", "cache": "warm", "allocation": False},
        "measurements": [
            {
                "workload": "small",
                "metric": "latency",
                "unit": "us",
                "direction": "lower",
                "value": value,
            }
        ],
    }
    result.update(changes)
    return result


def save(task, candidate, report=None, status="passed", parent=None, **kwargs):
    validation = f"{candidate}-validation.json"
    (task / validation).write_text(json.dumps({"status": status, "details": "task-owned result"}))
    path = None
    if report is not None:
        path = f"{candidate}-benchmark.json"
        (task / path).write_text(json.dumps(report))
    return evidence.record_candidate(
        task, candidate, ["solution/kernel.py"], validation, path, parent, **kwargs
    )


def test_cli_record_and_compare_from_another_directory(task, tmp_path):
    reports = [benchmark(2), benchmark(1)]
    reports[0]["measurements"].append(
        {
            "workload": "small",
            "metric": "throughput",
            "unit": "items/s",
            "direction": "higher",
            "samples": [12, 8, 10],
        }
    )
    reports[1]["measurements"].append(
        {
            "workload": "small",
            "metric": "throughput",
            "unit": "items/s",
            "direction": "higher",
            "value": 20,
        }
    )
    for name, report in zip(("base", "fast"), reports):
        (task / f"{name}.json").write_text(json.dumps(report))
        command = [
            sys.executable,
            str(SCRIPT),
            "record",
            "--workspace",
            str(task),
            "--candidate",
            name,
            "--source",
            "solution/kernel.py",
            "--validation",
            "validation.json",
            "--benchmark",
            f"{name}.json",
        ]
        if name == "fast":
            command += ["--parent", "base"]
        process = subprocess.run(
            command, cwd=task.parent, capture_output=True, text=True, check=True
        )
        assert Path(process.stdout.strip()).is_file()
    process = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "compare",
            "--workspace",
            str(task),
            "--baseline",
            "base",
            "--candidate",
            "fast",
            "--format",
            "json",
        ],
        cwd=task.parent,
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(process.stdout)
    candidate = report["candidates"][0]
    assert candidate["parent"] == "base"
    assert candidate["comparison_complete"]
    assert [row["speedup"] for row in candidate["measurements"]] == [2.0, 2.0]
    assert not any(row["regression"] for row in candidate["measurements"])


def test_records_keep_raw_reports_and_record_time_source_hashes(task):
    original = (task / "solution/kernel.py").read_bytes()
    report = benchmark(samples="unused extra metadata")
    path = save(task, "base", report)
    record = json.loads(path.read_text())
    raw = (task / "base-benchmark.json").read_bytes()
    assert (path.parent / "benchmark.json").read_bytes() == raw
    assert record["sources"] == {"solution/kernel.py": evidence._digest(original)}
    assert (task / "solution/kernel.py").read_bytes() == original
    (task / "base-benchmark.json").write_text("changed input report")
    (task / "solution/kernel.py").write_text("changed source after recording")
    save(task, "next", benchmark(1), parent="base")
    comparison = evidence.compare_candidates(task, "base", ["next"])
    assert comparison["candidates"][0]["measurements"][0]["speedup"] == 2
    assert comparison["baseline_sources"] != comparison["candidates"][0]["sources"]


def test_raw_samples_are_preserved_and_median_is_used(task):
    base, candidate = benchmark(), benchmark()
    for report, samples in [(base, [1, 100, 3, 5]), (candidate, [1, 3, 2])]:
        measurement = report["measurements"][0]
        del measurement["value"]
        measurement["samples"] = samples
    save(task, "base", base)
    path = save(task, "candidate", candidate)
    assert json.loads((path.parent / "benchmark.json").read_text()) == candidate
    row = evidence.compare_candidates(task, "base", ["candidate"])["candidates"][0]["measurements"][
        0
    ]
    assert row["baseline_value"] == 4
    assert row["candidate_value"] == 2
    assert row["speedup"] == 2


def test_regressions_and_missing_cases_remain_visible(task):
    base, candidate = benchmark(2), benchmark(4)
    base["measurements"].append(dict(base["measurements"][0], workload="baseline-only"))
    candidate["measurements"].append(dict(candidate["measurements"][0], workload="candidate-only"))
    save(task, "base", base)
    save(task, "candidate", candidate)
    report = evidence.compare_candidates(task, "base", ["candidate"])
    result = report["candidates"][0]
    assert not result["comparison_complete"]
    rows = {row["workload"]: row for row in result["measurements"]}
    assert rows["small"]["speedup"] == 0.5
    assert rows["small"]["regression"]
    assert rows["baseline-only"]["reasons"] == ["missing_candidate_measurement"]
    assert rows["candidate-only"]["reasons"] == ["missing_baseline_measurement"]
    assert rows["baseline-only"]["speedup"] is None
    assert "regression" in evidence.markdown_report(report)
    assert "missing_candidate_measurement" in evidence.markdown_report(report)


@pytest.mark.parametrize("field", ["environment", "protocol", "unit", "direction"])
def test_incompatible_measurements_have_no_speedup(task, field):
    base, candidate = benchmark(), benchmark(1)
    if field in ("environment", "protocol"):
        candidate[field] = {"different": "conditions"}
    else:
        candidate["measurements"][0][field] = "ms" if field == "unit" else "higher"
    save(task, "base", base)
    save(task, "candidate", candidate)
    result = evidence.compare_candidates(task, "base", ["candidate"])["candidates"][0]
    assert not result["comparison_complete"]
    assert f"{field}_mismatch" in result["measurements"][0]["reasons"]
    assert result["measurements"][0]["speedup"] is None


def test_protocol_comparison_distinguishes_boolean_from_number(task):
    base, candidate = benchmark(), benchmark(1)
    candidate["protocol"]["allocation"] = 0
    save(task, "base", base)
    save(task, "candidate", candidate)
    row = evidence.compare_candidates(task, "base", ["candidate"])["candidates"][0]["measurements"][
        0
    ]
    assert row["reasons"] == ["protocol_mismatch"]
    assert row["speedup"] is None


@pytest.mark.parametrize("failed", ["base", "candidate"])
@pytest.mark.parametrize("with_benchmark", [True, False])
def test_failed_validation_never_qualifies_for_speedup(task, failed, with_benchmark):
    for name in ("base", "candidate"):
        save(
            task,
            name,
            benchmark() if with_benchmark or name != failed else None,
            status="failed" if name == failed else "passed",
        )
    report = evidence.compare_candidates(task, "base", ["candidate"])
    result = report["candidates"][0]
    assert not result["comparison_complete"]
    assert (
        "baseline_validation_failed" if failed == "base" else "candidate_validation_failed"
    ) in result["issues"]
    assert all(row["speedup"] is None for row in result["measurements"])


def test_two_failed_candidates_without_benchmarks_render_a_status(task):
    save(task, "base", status="failed")
    save(task, "candidate", status="failed")
    report = evidence.compare_candidates(task, "base", ["candidate"])
    assert not report["candidates"][0]["comparison_complete"]
    assert "candidate_validation_failed" in evidence.markdown_report(report)


@pytest.mark.parametrize("bad", [True, None, "1", 0, -1, float("inf"), float("nan"), 10**400])
@pytest.mark.parametrize("samples", [False, True])
def test_invalid_measurements_are_rejected_before_record_creation(task, bad, samples):
    report = benchmark(bad)
    if samples:
        report["measurements"][0].pop("value")
        report["measurements"][0]["samples"] = [bad]
    with pytest.raises(ValueError):
        save(task, "invalid", report)
    assert not (task / evidence.STORE).exists()


@pytest.mark.parametrize(
    "edit", ["empty", "duplicate", "both", "empty_samples", "environment", "protocol", "direction"]
)
def test_malformed_benchmarks_are_rejected(task, edit):
    report = benchmark()
    if edit == "empty":
        report["measurements"] = []
    elif edit == "duplicate":
        report["measurements"] *= 2
    elif edit == "both":
        report["measurements"][0]["samples"] = [1]
    elif edit == "empty_samples":
        report["measurements"][0].pop("value")
        report["measurements"][0]["samples"] = []
    elif edit in ("environment", "protocol"):
        report[edit] = {}
    else:
        report["measurements"][0]["direction"] = "automatic"
    with pytest.raises(ValueError):
        save(task, "invalid", report)
    assert not (task / evidence.STORE).exists()


@pytest.mark.parametrize("status", [None, True, "skipped", "PASS"])
def test_validation_requires_explicit_passed_or_failed_status(task, status):
    with pytest.raises(ValueError, match="validation status"):
        save(task, "invalid", benchmark(), status=status)


def test_passed_validation_requires_benchmark(task):
    with pytest.raises(ValueError, match="require a benchmark"):
        save(task, "base")


@pytest.mark.parametrize("raw", [b'{"status":"passed","status":"failed"}', b"[]", b"{bad json"])
def test_ambiguous_or_malformed_reports_are_rejected(task, raw):
    (task / "validation.json").write_bytes(raw)
    with pytest.raises(ValueError):
        evidence.record_candidate(task, "base", ["solution/kernel.py"], "validation.json")
    assert not (task / evidence.STORE).exists()


def test_record_does_not_execute_reported_commands(task):
    target = task / "must-not-exist"
    (task / "validation.json").write_text(
        json.dumps({"status": "failed", "command": ["touch", str(target)]})
    )
    evidence.record_candidate(task, "base", ["solution/kernel.py"], "validation.json")
    assert not target.exists()


@pytest.mark.parametrize("name", ["../escape", "/absolute", ".", "", "a/b", "a" * 129])
def test_invalid_candidate_ids_are_rejected(task, name):
    with pytest.raises(ValueError):
        evidence.record_candidate(task, name, ["solution/kernel.py"], "validation.json")


def test_record_never_overwrites_existing_or_empty_candidate_directory(task):
    path = save(task, "base", benchmark())
    original = {file.name: file.read_bytes() for file in path.parent.iterdir()}
    with pytest.raises(ValueError, match="already exists"):
        save(task, "base", benchmark(100))
    assert {file.name: file.read_bytes() for file in path.parent.iterdir()} == original
    empty = task / evidence.STORE / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        save(task, "empty", benchmark())
    assert empty.is_dir() and not list(empty.iterdir())


def test_concurrent_record_creation_has_one_winner(task):
    (task / "benchmark.json").write_text(json.dumps(benchmark()))

    def record():
        try:
            evidence.record_candidate(
                task, "same", ["solution/kernel.py"], "validation.json", "benchmark.json"
            )
            return "created"
        except ValueError:
            return "exists"

    with ThreadPoolExecutor(max_workers=2) as executor:
        assert sorted(executor.map(lambda _: record(), range(2))) == ["created", "exists"]
    assert (task / evidence.STORE / "same/record.json").is_file()


def test_failed_write_cleans_only_the_new_record(task, monkeypatch):
    previous = save(task, "base", benchmark())
    original = Path.write_bytes

    def fail(path, data):
        if path.name == "benchmark.json" and path.parent.name == "broken":
            raise OSError("simulated write failure")
        return original(path, data)

    monkeypatch.setattr(Path, "write_bytes", fail)
    with pytest.raises(OSError, match="simulated"):
        save(task, "broken", benchmark())
    assert not (task / evidence.STORE / "broken").exists()
    assert previous.is_file()


def test_sources_must_be_distinct_regular_workspace_files(task):
    for sources in [
        ["../outside.py"],
        [str(task / "solution/kernel.py")],
        ["solution"],
        ["solution/kernel.py", "solution/kernel.py"],
    ]:
        with pytest.raises((ValueError, OSError)):
            evidence.record_candidate(task, "base", sources, "validation.json")
    (task / "alias").symlink_to(task / "solution", target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        evidence.record_candidate(task, "base", ["alias/kernel.py"], "validation.json")


def test_store_and_record_symlinks_are_rejected(task, tmp_path):
    outside = task / "outside"
    outside.mkdir()
    (task / evidence.STORE).symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        save(task, "base", benchmark())
    assert not list(outside.iterdir())


def test_parent_must_exist_and_cannot_be_self(task):
    with pytest.raises(OSError):
        save(task, "child", benchmark(), parent="missing")
    with pytest.raises(ValueError, match="own parent"):
        save(task, "self", benchmark(), parent="self")
    save(task, "base", benchmark())
    path = save(task, "child", benchmark(), parent="base")
    assert json.loads(path.read_text())["parent"] == "base"


@pytest.mark.parametrize("report", ["validation", "benchmark"])
def test_changed_stored_evidence_is_rejected(task, report):
    save(task, "base", benchmark())
    path = save(task, "candidate", benchmark())
    (path.parent / f"{report}.json").write_text("{}")
    with pytest.raises(ValueError, match="digest mismatch"):
        evidence.compare_candidates(task, "base", ["candidate"])


@pytest.mark.parametrize(
    "edit", ["schema", "missing_parent", "candidate", "source_digest", "reports"]
)
def test_invalid_manifests_fail_closed(task, edit):
    save(task, "base", benchmark())
    path = save(task, "candidate", benchmark())
    manifest = json.loads(path.read_text())
    if edit == "schema":
        manifest["schema_version"] = True
    elif edit == "missing_parent":
        del manifest["parent"]
    elif edit == "candidate":
        manifest["candidate"] = "other"
    elif edit == "source_digest":
        manifest["sources"]["solution/kernel.py"] = "not-a-digest"
    else:
        manifest["reports"] = {}
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        evidence.compare_candidates(task, "base", ["candidate"])


def test_markdown_escapes_cells_and_comparison_order_is_stable(task):
    report = benchmark()
    report["measurements"][0]["workload"] = "a|b\n<script>"
    save(task, "base", report)
    save(task, "z", report)
    save(task, "a", report)
    first = evidence.compare_candidates(task, "base", ["z", "a"])
    assert first == evidence.compare_candidates(task, "base", ["a", "z"])
    markdown = evidence.markdown_report(first)
    assert "a&#124;b &lt;script&gt;" in markdown
    assert "<script>" not in markdown
    assert "not an independent correctness certification" in markdown


def test_unrepresentable_ratio_is_not_reported_as_speedup(task):
    save(task, "base", benchmark(1e300))
    save(task, "candidate", benchmark(1e-300))
    result = evidence.compare_candidates(task, "base", ["candidate"])
    row = result["candidates"][0]["measurements"][0]
    assert row["reasons"] == ["unrepresentable_speedup"]
    assert row["speedup"] is None
    json.dumps(result, allow_nan=False)


def test_cli_errors_are_actionable_and_do_not_print_tracebacks(task):
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "compare",
            "--workspace",
            str(task),
            "--baseline",
            "missing",
            "--candidate",
            "other",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "Evidence error:" in result.stderr
    assert "Traceback" not in result.stderr
