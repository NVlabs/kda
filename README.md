# KDA Wishlist

Submit reproducible kernel requests for Kernel Design Agents (KDA).

[Open a request](https://github.com/NVlabs/kda/issues/new?template=kernel-request.yml) · [Browse requests](https://github.com/NVlabs/kda/issues?q=is%3Aissue%20is%3Aopen%20label%3Awishlist) · [Example](example/)

## Structure

```text
README.md
example/
  README.md
  definition.json
  workloads.jsonl
  baseline.py
requests/
  <issue-number>-<kernel-name>/
    README.md
    definition.json
    workloads.jsonl
    baseline.py
    benchmark.py       # optional
```

| File | Purpose |
| --- | --- |
| `README.md` | Issue link, project impact, target hardware, environment, commands, and baseline source/license. |
| `definition.json` | [FlashInfer Trace](https://bench.flashinfer.ai/docs/flashinfer-trace) definition, including the mathematical reference implementation for correctness. |
| `workloads.jsonl` | Workload traces: one JSON object per line, referencing the definition. |
| `baseline.py` | The current best-known implementation (state of the art, or SOTA) to compare against. Record its source and version in the README. |
| `benchmark.py` | Optional custom benchmark. If absent, use `flashinfer-bench`, as shown in the example. |

`baseline.py` contains the implementation, not the benchmark driver. Expose a `run` function with the same input order and returned outputs as the definition's reference. Put custom timing logic in the optional `benchmark.py`; document commands and measurement settings in the request README.

## Naming and Collaboration

Use `requests/<issue-number>-<kernel-name>/`, for example `requests/123-rmsnorm/`. The number must be an issue in `NVlabs/kda`; use lowercase letters, digits, and hyphens for the short kernel name.

Each issue has one directory. Different contributors' requests are distinguished by issue number, even when the kernels have the same name. To add workloads or improve the baseline for an existing request, submit changes to its existing directory. Keep directory names stable after merging, and coordinate shared-file changes through the issue and pull request.

Use a distinct definition name and fresh workload identifiers when copying the example. Every workload's `definition` value must match its definition's `name`.

## Submit

1. Copy `example/` into your fork and replace it with your request materials. Commit a draft so you have a public, fixed-version definition link for the issue form.
2. Open a **Kernel wishlist** issue. Its title defaults to `[wishlist] `. Select NVIDIA B200, NVIDIA B300, or both; these are the currently supported GPUs.
3. Name the final directory using the assigned issue number and update the issue's material link to the final commit.
4. Document the contract, correctness tolerances, representative workloads, baseline source/version, dependencies, licenses, and reproduction commands. Include actual validation results for your selected hardware.
5. Sign off commits with `git commit -s` and open a pull request targeting **`wishlist`**, linking the issue.

If `benchmark.py` is present, use its documented command. Otherwise evaluate the definition, workloads, and baseline with `flashinfer-bench`; the [example README](example/README.md) provides the complete command. Do not add a separate `reproduce.py`.

Follow the [contribution process](https://github.com/NVlabs/kda/blob/main/CONTRIBUTING.md) and [project license](https://github.com/NVlabs/kda/blob/main/LICENSE). First-party code uses Apache-2.0 and documentation uses Creative Commons Attribution 4.0; preserve the licenses and attribution of any third-party material. Keep generated results, caches, and large datasets outside this branch.

Issue forms are maintained on [`main`](https://github.com/NVlabs/kda/tree/main/.github/ISSUE_TEMPLATE). Website source is on [`pages`](https://github.com/NVlabs/kda/tree/pages).
