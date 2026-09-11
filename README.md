# KDA Wishlist

KDA Wishlist is a community program for turning reproducible kernel definitions into optimized kernel solutions. A kernel is a performance-critical program that runs on accelerated hardware, such as a graphics processing unit (GPU), and implements the core computation of a machine learning operation.

Live site: <https://nvlabs.github.io/kda/>

KDA stands for Kernel Design Agents: a workflow in which coding agents research, implement, verify, profile, and iterate on performance-sensitive kernel tasks.

This website introduces the program and directs contributors with prepared files to submit a pull request (PR) to `wishlist`. The main **Submit a request** link opens the submission guide; **Discuss an idea** opens an optional GitHub issue for early discussion or help preparing the files. Request details live in the submitted README, so contributors do not have to repeat them in an issue or PR description.

## Run locally

The project requires Node.js 22.13.0 or later.

Install the dependencies and start the development server:

```bash
npm install
npm run dev
```

Then open `http://localhost:3000`.

To run the production build locally:

```bash
npm run build
npm run start -- --port 3000
```

To create the static export used by GitHub Pages:

```bash
GITHUB_PAGES=true GITHUB_REPOSITORY=NVlabs/kda SITE_URL=https://nvlabs.github.io/kda npm run build:pages
```

The exported site is written to `out/`.

## Validate the project

```bash
npm run build
npm run lint
npm audit
```

The production build targets Cloudflare Workers, Cloudflare's distributed edge runtime. The logical hosting configuration lives in `.openai/hosting.json`.

Pushes to `pages` run `.github/workflows/deploy-pages.yml`, which builds the static export and deploys it to GitHub Pages.

## Submission workflow

1. Follow the [submission guide](https://github.com/NVlabs/kda/tree/wishlist#submit). Create a branch from `wishlist` in your fork and copy `example/` to `requests/<github-username>-<kernel-name>/`. Define the task in FlashInfer Trace, a reproducible format that describes the reference implementation, input and output contract, correctness requirements, and representative workloads. Include the best-known comparison implementation in `baseline.py`; `benchmark.py` is optional, with `flashinfer-bench` used when it is absent.
2. Submit a PR directly to `NVlabs/kda:wishlist`; no issue or assigned number is needed. Keep the request details in its `README.md` and the PR description brief. Currently, KDA only supports NVIDIA B200 and B300 GPUs. If you need help preparing the materials, [discuss an idea](https://github.com/NVlabs/kda/issues/new?template=kernel-request.yml); complete files are optional for that discussion.
3. Community members add thumbs-up reactions to the PR description and use comments to contribute new workload evidence or implementation context.
4. The team reviews the task. Merging records the request; it does not mean optimization is complete. Progress and result links remain on the original PR after merging. Selected requests enter a measured loop of research, implementation, correctness validation, performance profiling, and candidate selection.
5. Completed tasks may return an optimized kernel, benchmark comparisons, reproduction instructions, environment details, design notes, known limitations, and an upstream-ready contribution.

[Browse and upvote wishlist requests](https://github.com/NVlabs/kda/pulls?q=is%3Apr%20base%3Awishlist). The list includes open and closed PRs so collected requests remain visible without requiring a label first.

Submission does not guarantee selection. The program prioritizes tasks that affect real systems, can be evaluated automatically, produce publicly reproducible results, benefit multiple projects, and have a realistic path to upstream adoption.

## Repository structure

- `app/` contains the page structure, copy, metadata, and styles.
- `public/og.png` is the branded social-sharing preview image.
- `.github/workflows/deploy-pages.yml` builds and deploys this `pages` branch.
- [Issue forms and settings](https://github.com/NVlabs/kda/tree/main/.github/ISSUE_TEMPLATE) are maintained on the default `main` branch so GitHub can load them.
- [The `wishlist` branch](https://github.com/NVlabs/kda/tree/wishlist) contains submission instructions and contributed request materials.
- `.openai/hosting.json` contains the logical website-hosting configuration.

## References

- [Kernel Design Agents workflow](https://github.com/NVlabs/kda)
- [MLSys 2026 FlashInfer contest](https://mlsys26.flashinfer.ai/)
- [Contest workflow and results release](https://github.com/mit-han-lab/mlsys2026-flashinfer-contest)
- [FlashInfer Bench and the Trace format](https://github.com/flashinfer-ai/flashinfer-bench)
