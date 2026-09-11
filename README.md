# Kernel Design Agents

Kernel Design Agents (KDA) is an agent-centric workflow for using coding agents to research, implement, verify, and iterate on performance-sensitive CUDA kernel tasks.

This repository documents an early research prototype and remains under active development. Community feedback and contributions are welcome. If you are interested in HAN Lab Mafia's solutions ranking #1–3 on tracks at the MLSys Kernel Contest, see [mit-han-lab/mlsys2026-flashinfer-contest](https://github.com/mit-han-lab/mlsys2026-flashinfer-contest) for performance evaluation and reproduction.

## Contents

| Path | Purpose |
|---|---|
| `docs/agent-flow.md` | Minimal end-to-end KDA workflow. |
| `prompts/README.md` | How to use prompt templates. |
| `prompts/basic-flow.md` | Generic starter prompt for a new task. |
| `CLAUDE.md` | Repository-facing agent instructions. |
| `CONTRIBUTING.md` | Contribution process and DCO sign-off requirements. |
| `THIRD_PARTY_NOTICES.md` | Third-party component and license disclosures. |
| `third_party_licenses/` | Verbatim upstream license files for distributed third-party components. |

## Community Kernel Wishlist

Have a kernel that needs optimization? [Submit a request](https://github.com/NVlabs/kda/tree/wishlist#submit) directly as a pull request to `wishlist`, with a reproducible definition, representative workloads, and your best-known baseline implementation. No issue is required. The community wishlist currently supports NVIDIA B200 and B300 GPUs only.

[Browse and upvote requests](https://github.com/NVlabs/kda/pulls?q=is%3Apr%20base%3Awishlist), [discuss an idea](https://github.com/NVlabs/kda/issues/new?template=kernel-request.yml) if you need help preparing the files, or visit the [project website](https://nvlabs.github.io/kda/). Keep request details in `requests/<github-username>-<kernel-name>/README.md`; the pull request description can be brief. Merging records the request, while optimization progress and result links remain on the original pull request. Website source lives on `pages`; active issue forms are maintained on `main`.

## Getting Started
Install the agent workflow dependencies before starting the agent session:

```bash
git clone --recurse-submodules https://github.com/mit-han-lab/kernel-design-agents.git
cd kernel-design-agents

# link skills
mkdir -p ~/.claude/skills
ln -s "$(pwd)/skills/ncu-report-skill" ~/.claude/skills/ncu-report-skill
ln -s "$(pwd)/skills/KernelWiki" ~/.claude/skills/KernelWiki

# or clone the independently licensed ncu skill directly
mkdir -p ~/.claude/skills && cd ~/.claude/skills
git clone https://github.com/mit-han-lab/ncu-report-skill.git
```

Use KernelWiki from this repository's pinned submodule. A direct upstream
checkout may contain artifact snapshots governed by additional terms that are
omitted from this distribution.

## Submodules

The following third-party projects are included as Git submodules. The source URLs and revisions below are intentional and match the gitlinks recorded in this repository.

| Path | Source | Pinned revision | License |
|---|---|---|---|
| `skills/KernelWiki` | [mit-han-lab/KernelWiki](https://github.com/mit-han-lab/KernelWiki.git) | Based on [`76d27b56f804e7e7295d4c570e1e5d7eef4b0a75`](https://github.com/mit-han-lab/KernelWiki/commit/76d27b56f804e7e7295d4c570e1e5d7eef4b0a75), with the repository-maintained release sanitization recorded by the gitlink | MIT for original material; embedded artifacts retain their upstream terms; restricted CuTe DSL artifacts are omitted |
| `skills/ncu-report-skill` | [mit-han-lab/ncu-report-skill](https://github.com/mit-han-lab/ncu-report-skill.git) | [`d1887948c7d53690cfe6605f59c1329b8a1c6bb5`](https://github.com/mit-han-lab/ncu-report-skill/commit/d1887948c7d53690cfe6605f59c1329b8a1c6bb5) | MIT |

These submodules are not covered by this repository's first-party license. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the applicable license text and additional details.

Install the `humanize` Claude Code plugin from the Claude Code plugin UI:

```text
/plugin marketplace add PolyArch/humanize
/plugin install humanize@PolyArch
```

## Minimal Flow

1. Create a separate implementation workspace for the target task.
2. Define the task contract: objective, constraints, validation command, and promotion criteria.
3. Start an agent session in the implementation workspace.
4. Give the agent `prompts/basic-flow.md`, filled in with the task-specific details.
5. Ask the agent to write a short plan draft to `docs/draft.md` in the implementation workspace.
6. Convert the draft into an executable plan, either manually or with a planning tool such as Humanize.
7. Implement in small iterations, verifying after each meaningful change.
8. Record candidates, benchmark or evaluation results, profiling evidence, and final promotion decisions.

The workflow is intentionally independent of any single benchmark harness or hardware target. A downstream task can add its own evaluator, datasets, profiling tools, and domain-specific references.

## Recommended Workspace Layout

Use this repository as reference material, then do implementation work elsewhere:

```text
task-workspace/
  docs/
    draft.md
    plan.md
  runs/
  outputs/
  profile/
  benchmark.csv
  candidates.jsonl
```

The exact files can change by domain. The important rule is that the agent records enough context for another engineer to understand what was tried, what passed validation, and why the final candidate was selected.

## Contributing

This project accepts outside contributions under the Developer Certificate of Origin process. See [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

## License

Except for the third-party submodules identified above, first-party documentation, prompts, skills-style content, and assets are licensed under the [Creative Commons Attribution 4.0 International License](LICENSE), and first-party source code is licensed under the [Apache License 2.0](LICENSE).
