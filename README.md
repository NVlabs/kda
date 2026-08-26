# Kernel Design Agents

Kernel Design Agents (KDA) is a agent-centric workflow for using coding agents to research, implement, verify, and iterate on performance-sensitive CUDA kernel tasks.

This repository documents the early research prototype and is still under active development (we are looking for community feedbacks!). If you are interested in HAN Lab Mafia's  solution ranking #1~3 on tracks at MLSys Kernel Contest, please refer to [mit-han-lab/mlsys2026-flashinfer-contest](https://github.com/mit-han-lab/mlsys2026-flashinfer-contest) for perform evaluation and reproducement.

## Contents

| Path | Purpose |
|---|---|
| `docs/agent-flow.md` | Minimal end-to-end KDA workflow. |
| `prompts/README.md` | How to use prompt templates. |
| `prompts/basic-flow.md` | Generic starter prompt for a new task. |
| `CLAUDE.md` | Repository-facing agent instructions. |

## Getting Started
Install the agent workflow dependencies before starting the agent session:

```bash
git clone --recurse-submodules https://github.com/mit-han-lab/kernel-design-agents.git
cd kernel-design-agents

# link skills
mkdir -p ~/.claude/skills
ln -s "$(pwd)/skills/ncu-report-skill" ~/.claude/skills/ncu-report-skill
ln -s "$(pwd)/skills/KernelWiki" ~/.claude/skills/KernelWiki

# or clone skills directly
mkdir -p ~/.claude/skills && cd ~/.claude/skills
git clone https://github.com/mit-han-lab/ncu-report-skill.git
git clone https://github.com/mit-han-lab/KernelWiki.git
```

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

## License

This project is licensed under the [Apache License 2.0](LICENSE).
