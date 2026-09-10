# KDA Wishlist

KDA Wishlist is a community program for turning reproducible kernel definitions into optimized kernel solutions. A kernel is a performance-critical program that runs on accelerated hardware, such as a graphics processing unit (GPU), and implements the core computation of a machine learning operation.

Live site: <https://nvlabs.github.io/kda/>

KDA stands for Kernel Design Agents: a workflow in which coding agents research, implement, verify, profile, and iterate on performance-sensitive kernel tasks.

This website introduces the program, helps contributors prepare reproducible kernel definitions and workloads, and directs them to GitHub Issues. GitHub Issues is the repository's public request tracker, where contributors submit kernel needs, add useful context, and signal demand with thumbs-up reactions.

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

1. Define the task in FlashInfer Trace, a reproducible format that describes the reference implementation, input and output contract, correctness requirements, and representative workloads.
2. Open a GitHub issue with the [Kernel wishlist form](https://github.com/NVlabs/kda/issues/new?template=kernel-request.yml). Currently, KDA only supports NVIDIA B200 and B300 GPUs. Follow the [submission guide](https://github.com/NVlabs/kda/tree/wishlist); material pull requests target `wishlist`.
3. Community members add thumbs-up reactions to the top-level issue and use comments to contribute new workload evidence or implementation context.
4. The team reviews the task. Selected requests enter a measured loop of research, implementation, correctness validation, performance profiling, and candidate selection.
5. Completed tasks may return an optimized kernel, benchmark comparisons, reproduction instructions, environment details, design notes, known limitations, and an upstream-ready contribution.

[Browse and upvote wishlist requests](https://github.com/NVlabs/kda/issues?q=is%3Aissue%20is%3Aopen%20label%3Awishlist).

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
