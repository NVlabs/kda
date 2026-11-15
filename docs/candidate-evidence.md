# Record and compare candidate evidence

The optional `scripts/candidate_evidence.py` utility imports existing task reports
and compares candidates with a named baseline. It uses only the Python standard
library (Python 3.11+); it does not need a GPU or execute candidate code.

Keep the task's source, validators, evaluators and evidence in a separate task
workspace. Existing manually maintained records remain valid; using this utility
is not a requirement for the KDA loop.

## Prepare task-owned reports

Run the task's validator and evaluator first. Adapt their results to the following
small JSON formats. The adapter belongs to the task, not KDA's generic repository.

A validation report, for example `runs/base-validation.json`:

```json
{
  "status": "passed",
  "command": ["python", "validate.py"],
  "details": "All task-defined cases passed."
}
```

`status` must be `passed` or `failed`. Other fields are preserved as supplied.
They are not executed or independently checked. In particular, a reported `passed`
status is not a correctness certification by this utility.

A benchmark report, for example `runs/base-benchmark.json`:

```json
{
  "environment": {"device": "target-device", "runtime": "tested-version"},
  "protocol": {"timing": "cuda-graph", "cache": "warm", "allocation": "excluded"},
  "measurements": [
    {
      "workload": "case-1",
      "metric": "latency",
      "unit": "us",
      "direction": "lower",
      "samples": [10, 12, 14]
    }
  ]
}
```

The numbers above illustrate the format, not measured performance.

- `environment` and `protocol` must be nonempty JSON objects. Include enough
  information to make comparisons meaningful: hardware, relevant library/compiler
  versions, timing scope, input/cache policy, warmup and repeat policy, as relevant.
  The tool does not inspect devices, collect environment variables or infer these
  conditions. Extra fields in only one report intentionally cause a mismatch.
- Each measurement declares nonempty `workload`, `metric` and `unit` strings,
  plus `direction` (`lower` for latency, `higher` for throughput).
- Identify a workload completely, including shapes, dtype, layout and any other
  input characteristic needed by the task. A `(workload, metric)` pair must be unique.
- Provide exactly one of `value` or `samples`. Numbers must be positive and finite;
  `samples` must be a nonempty list. Samples are preserved and their median is used
  for comparison. The utility does not estimate statistical significance.
- Units are explicit labels and are never converted. Use matching labels for
  comparable measurements. Do not substitute a single launch timing for an entire
  API call or a warm-cache measurement for a cold-cache one.
- JSON objects with duplicate keys and non-finite numbers are rejected.

A failed candidate may omit its benchmark report. A passing candidate must supply
one. A failed candidate with available measurements can retain those measurements,
but they will not receive a speedup in the comparison.

## Record a baseline and a candidate

From the KDA checkout, assuming the task has already produced both reports and
`solution/kernel.py`:

```sh
python scripts/candidate_evidence.py record \
  --workspace ../task-workspace \
  --candidate base \
  --source solution/kernel.py \
  --validation runs/base-validation.json \
  --benchmark runs/base-benchmark.json
```

After implementing and validating a new candidate, record it under a new ID:

```sh
python scripts/candidate_evidence.py record \
  --workspace ../task-workspace \
  --candidate grouped --parent base \
  --source solution/kernel.py --source solution/helper.py \
  --validation runs/grouped-validation.json \
  --benchmark runs/grouped-benchmark.json
```

Repeat `--source` for all relevant source files; selecting the complete input
set is the task's responsibility. Sources and reports must be regular files with
relative paths inside the workspace; symlinks and parent traversal are rejected.
A supplied parent must already have a valid record. IDs are 1-128 characters,
start with a letter or digit, and otherwise use letters, digits, `.`, `_` or `-`.

The tool creates `candidate-evidence/<ID>/` in the task workspace. It contains:

- `record.json`: schema version, candidate/parent IDs, UTC recording time, selected
  source SHA256 digests, and raw-report digests;
- `validation.json`: an exact copy of the supplied validation report;
- `benchmark.json`: an exact copy of the benchmark report, when supplied.

Source files are hashed, not copied or changed. Digests describe the selected
files **at recording time**. They do not prove that those exact bytes were used by
an earlier validation/benchmark run, nor cover unselected dependencies. Record
promptly after evaluating the candidate and preserve task-specific provenance in
the reports. A task that needs stronger provenance should bind its evaluator to
its own source snapshots.

An existing candidate directory is never overwritten, even if empty. Use a new ID
for each new run. Report copies are written before an atomic publication of
`record.json`. Comparisons reject incomplete records. A handled write failure
removes only the new record directory; process termination or machine failure can
leave an incomplete directory for manual inspection. These are local evidence
records, not a signed or tamper-proof store.

## Compare without promoting

```sh
python scripts/candidate_evidence.py compare \
  --workspace ../task-workspace --baseline base \
  --candidate grouped --format markdown
```

Use `--format json` for machine-readable output and repeat `--candidate` to
compare multiple candidates. Results go to standard output, so callers choose
where to save them. The utility does not modify stored records during comparison.
Candidate IDs and workload/metric keys are sorted for deterministic output.

For each workload/metric, the report shows values, units, a speedup when eligible,
and any incompatibility reasons. Speedup is baseline/candidate for `lower`, and
candidate/baseline for `higher`; below 1 is a regression.

A speedup is omitted if either reported validation failed, either measurement is
missing, units/directions differ, or the environment/protocol objects differ.
Metadata comparison is exact (ignoring object-key order), with no automatic
normalization. Missing cases on either side remain visible. Corrupt records or
changed stored report bytes cause an explicit error; source digests are not
rechecked against the current, potentially edited task sources.

`comparison_complete` means every measurement in the union was comparable. It
does not mean the candidate improved or should be accepted. There is no aggregate
score, significance test or automatic promotion. Promotion remains governed by
the task contract. Review regressions and all remaining evidence before selecting
a candidate.

Exit status 0 means the record or report was generated successfully, including a
report showing failed validation or incompatible measurements. Invalid inputs,
missing/corrupt records and I/O failures exit with status 1 and an explanatory
message. Command-line usage errors use argparse's exit status 2. An incompatible
report is not a validation pass.
