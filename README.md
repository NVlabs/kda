# KDA Wishlist

Submit reproducible kernel requests for Kernel Design Agents (KDA).

[Submit a request](#submit) · [Browse requests](https://github.com/NVlabs/kda/pulls?q=is%3Apr%20base%3Awishlist) · [Discuss an idea](https://github.com/NVlabs/kda/issues/new?template=kernel-request.yml) · [Example](example/)

Have the files ready? Submit a pull request (PR) directly to `wishlist`. No issue is required. If you need help shaping the request, use **Discuss an idea** first; complete files are optional for that discussion.

## Structure

```text
README.md
example/
  README.md
  definition.json
  workloads.jsonl
  baseline.py
requests/
  <github-username>-<kernel-name>/
    README.md
    definition.json
    workloads.jsonl
    baseline.py
    benchmark.py       # optional
```

| File | Purpose |
| --- | --- |
| `README.md` | Project impact, target hardware, environment, commands, validation results, and baseline source/license. Related issue links are optional. |
| `definition.json` | [FlashInfer Trace](https://bench.flashinfer.ai/docs/flashinfer-trace) definition, including the mathematical reference implementation for correctness. |
| `workloads.jsonl` | Workload traces: one JSON object per line, referencing the definition. |
| `baseline.py` | The current best-known implementation (state of the art, or SOTA) to compare against. Record its source and version in the README. |
| `benchmark.py` | Optional custom benchmark. If absent, use `flashinfer-bench`, as shown in the example. |

`baseline.py` contains the implementation, not the benchmark driver. Expose a `run` function with the same input order and returned outputs as the definition's reference. Put custom timing logic in the optional `benchmark.py`; document commands and measurement settings in the request README.

## Naming and Collaboration

Use `requests/<github-username>-<kernel-name>/`, for example `requests/alice-rmsnorm/`. Lowercase your GitHub username and use lowercase letters, digits, and hyphens for the short kernel name. You can name the directory before opening a PR; no issue or PR number is needed.

Different contributors can submit the same kernel name under their own usernames. If you have a separate request for a variant of the same kernel, add a descriptive suffix, such as `alice-rmsnorm-backward`. Check existing requests before adding a new one. To add workloads or improve the baseline for an existing request, edit its existing directory regardless of who first submitted it. Keep directory names stable after merging, including if the original contributor changes their username.

Use a distinct definition name and fresh workload identifiers when copying the example. Every workload's `definition` value must match its definition's `name`.

## Submit

1. [Fork the repository](https://github.com/NVlabs/kda/fork). Clear **Copy the main branch only** so your fork includes `wishlist`, then create your working branch from `wishlist`.
2. Copy `example/` to `requests/<github-username>-<kernel-name>/` and replace the example with your request. Document the project impact, contract, correctness tolerances, representative workloads, baseline source/version, dependencies, licenses, and commands in its `README.md`. Target NVIDIA B200, NVIDIA B300, or both, and include actual validation results for the selected hardware.
3. Commit and push the files to your fork, signing off each commit with `git commit -s`. You can also upload a prepared `requests/` folder from your working branch's repository root using GitHub's **Add file → Upload files**. For web commits, add `Signed-off-by: Your Name <your@email.com>` to the commit description, following the [contribution process](https://github.com/NVlabs/kda/blob/main/CONTRIBUTING.md#signing-off-your-work).
4. Open a PR with **base repository: `NVlabs/kda`**, **base: `wishlist`**, and your fork's working branch as the head. Use a title such as `[wishlist] RMSNorm for <project>`. Keep the description short and point to your request directory; the files already contain the details. Link an existing discussion if relevant, but you do not need to create an issue.

Check that the PR contains only the intended request files. Respond to review by pushing updates to the same branch; the PR updates automatically.

If `benchmark.py` is present, use its documented command. Otherwise evaluate the definition, workloads, and baseline with `flashinfer-bench`; the [example README](example/README.md) provides the complete command. Do not add a separate `reproduce.py`.

## Review and Progress

Maintainers review the files and use the `wishlist`, `needs-triage`, `needs-info`, `accepted`, and `in-progress` labels to track requests. Community members can add a thumbs-up reaction to the PR description and contribute workload evidence in comments.

Merging a PR records the request in the wishlist; it does not mean an optimized kernel has been delivered. Maintainers post progress and result links on the original PR, including after it is merged. [Browse requests](https://github.com/NVlabs/kda/pulls?q=is%3Apr%20base%3Awishlist) includes both open and closed PRs so collected requests remain visible.

Follow the [contribution process](https://github.com/NVlabs/kda/blob/main/CONTRIBUTING.md) and [project license](https://github.com/NVlabs/kda/blob/main/LICENSE). First-party code uses Apache-2.0 and documentation uses Creative Commons Attribution 4.0; preserve the licenses and attribution of any third-party material. Keep generated results, caches, and large datasets outside this branch.

Issue forms are maintained on [`main`](https://github.com/NVlabs/kda/tree/main/.github/ISSUE_TEMPLATE). Website source is on [`pages`](https://github.com/NVlabs/kda/tree/pages).
