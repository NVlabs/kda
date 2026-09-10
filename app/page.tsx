const NEW_REQUEST_URL =
  'https://github.com/humanfia/KDA-wishlist/issues/new?template=kernel-request.yml';
const ISSUES_URL = 'https://github.com/humanfia/KDA-wishlist/issues';
const FLASHINFER_TRACE_URL =
  'https://bench.flashinfer.ai/docs/flashinfer-trace';
const TECH_REPORT_URL =
  'https://github.com/mit-han-lab/mlsys2026-flashinfer-contest/blob/main/docs/HAN_Lab_Kernel_Mafia_Technical_Report.pdf';
const CONTEST_SOLUTIONS_URL =
  'https://github.com/mit-han-lab/mlsys2026-flashinfer-contest-solution';
const KDA_01_CONTEST_SOLUTIONS_URL =
  'https://github.com/mit-han-lab/mlsys2026-flashinfer-contest-solution/tree/afef448b7d22f3d20128e87ab9afe3e565cb5812';

const achievements = [
  {
    metric: '1.39×',
    metricLabel: 'peak vs. human SOTA',
    title: 'KDA-0.5 surpassed human SOTA',
    benchmark: 'MLSys’26 FlashInfer Kernel Contest · B300 · MoE 2.25× · DSA 29.95× · GDN 6.10×',
    id: 'kda-1.5-flashinfer-contest',
    project: 'FLASHINFER CONTEST',
    status: 'SOTA',
    linkLabel: 'Contest solution repo',
    href: CONTEST_SOLUTIONS_URL,
  },
  {
    metric: '#1',
    metricLabel: 'SOLExec L1',
    title: 'KDA boosts Databricks to rank #1 on SOL-Bench',
    benchmark: 'L1 Single Operation track · score 0.7608 · previous best 0.7584',
    id: 'solexec-l1',
    project: 'SOLEXEC',
    status: '1ST PLACE',
    linkLabel: 'Announcement on X',
    href: 'https://x.com/Yuchenj_UW/status/2072496816336290143',
  },
  {
    metric: 'TOP3',
    metricLabel: 'all contest tracks',
    title: 'KDA-0.1 achieves TOP3 on every track',
    benchmark: 'MLSys’26 FlashInfer Contest · B300 · MoE 0.67× · DSA 11.91× · GDN 1.16×',
    id: 'kda-0.1-flashinfer-contest',
    project: 'FLASHINFER CONTEST',
    status: 'TOP 3',
    linkLabel: 'MLSys26 FlashInfer Contest',
    href: 'https://mlsys26.flashinfer.ai/',
  },
  {
    metric: '6.1×',
    metricLabel: 'balanced k-means',
    title: 'Flash-KMeans',
    benchmark: 'KDA speedup for the Wan 2.2 workload on B200',
    id: 'flash-kmeans',
    project: 'FLASH-KMEANS',
    status: 'OPEN SOURCE',
    linkLabel: 'Flash-KMeans #23',
    href: 'https://github.com/svg-project/flash-kmeans/pull/23',
  },
  {
    metric: '2.03×',
    metricLabel: 'weighted B200 kernel group',
    title: 'Cosmos3 causal Conv3D cat-pad',
    benchmark: '10.621 → 5.240 ms weighted; model E2E +2.11%',
    id: '29281',
    project: 'SGLANG',
    status: 'MERGED',
    linkLabel: 'SGLang #29281',
    href: 'https://github.com/sgl-project/sglang/pull/29281',
  },
  {
    metric: '5.84×',
    metricLabel: 'kernel geomean',
    title: 'LTX-2.3 QKNorm + split-RoPE',
    benchmark: 'B200 · 4.22–7.34× across 14 production shapes; E2E −9.16%',
    id: '29708',
    project: 'SGLANG',
    status: 'MERGED',
    linkLabel: 'SGLang #29708',
    href: 'https://github.com/sgl-project/sglang/pull/29708',
  },
] as const;

const moreImprovements = [
  {
    result: '1.21× selected GEMM',
    operator: 'KDA NVFP4 GEMM for Qwen3.8 on SM120',
    repoLabel: 'SGLang #36865',
    href: 'https://github.com/sgl-project/sglang/pull/36865',
  },
  {
    result: '+4.45% E2E throughput',
    operator: 'Restore SM121 QSA correctness with Humanize + KDA',
    repoLabel: 'SGLang #36845',
    href: 'https://github.com/sgl-project/sglang/pull/36845',
  },
  {
    result: '4.24–19.86% faster',
    operator: 'Blackwell fused MHC half-MMA kernel',
    repoLabel: 'TensorRT-LLM #16799',
    href: 'https://github.com/NVIDIA/TensorRT-LLM/pull/16799',
  },
  {
    result: '1.06× peak GEMM speedup',
    operator: 'SM100 NVFP4 SVDQuant fused GEMM',
    repoLabel: 'FlashInfer #3858',
    href: 'https://github.com/flashinfer-ai/flashinfer/pull/3858',
  },
  {
    result: '5.84× kernel geomean',
    operator: 'LTX-2.3 QKNorm + split-RoPE CUDA fast path',
    repoLabel: 'SGLang #29708',
    href: 'https://github.com/sgl-project/sglang/pull/29708',
  },
  {
    result: '1.032× model E2E',
    operator: 'LTX-2 residual-gate CUDA fast path',
    repoLabel: 'SGLang #29361',
    href: 'https://github.com/sgl-project/sglang/pull/29361',
  },
  {
    result: '2.03× weighted kernel group',
    operator: 'Cosmos3 causal Conv3D cat-pad CUDA fast path',
    repoLabel: 'SGLang #29281',
    href: 'https://github.com/sgl-project/sglang/pull/29281',
  },
  {
    result: '1.06× denoise speedup',
    operator: 'Sana packed self- and cross-attention GEMMs',
    repoLabel: 'SGLang #28393',
    href: 'https://github.com/sgl-project/sglang/pull/28393',
  },
  {
    result: '1.015× denoise speedup',
    operator: 'FLUX fused FeedForward GELU up-projection',
    repoLabel: 'SGLang #28166',
    href: 'https://github.com/sgl-project/sglang/pull/28166',
  },
  {
    result: '1.13× denoise speedup',
    operator: 'Qwen-Image norm-scale-shift',
    repoLabel: 'SGLang #27392',
    href: 'https://github.com/sgl-project/sglang/pull/27392',
  },
  {
    result: '1.06× chat throughput',
    operator: 'Qwen3-Next FP8 MoE',
    repoLabel: 'SGLang #26204',
    href: 'https://github.com/sgl-project/sglang/pull/26204',
  },
] as const;

const workflow = [
  {
    number: '01',
    title: 'Open a request',
    copy: 'Share a reproducible FlashInfer Trace definition and explain why the kernel matters.',
    teaser: 'request',
  },
  {
    number: '02',
    title: 'Community votes',
    copy: 'Others add a thumbs-up reaction to the GitHub issue and contribute useful workload context.',
    teaser: 'signals',
  },
  {
    number: '03',
    title: 'KDA thinks and builds',
    copy: 'Accepted tasks enter a measured loop of analysis, generation, correctness checks, and profiling.',
    teaser: 'agents',
  },
  {
    number: '04',
    title: 'Results return',
    copy: 'We publish the strongest validated kernel with benchmarks, reproduction steps, and known limits.',
    teaser: 'results',
  },
] as const;

type WorkflowTeaserKind = (typeof workflow)[number]['teaser'];

function WorkflowTeaser({ kind }: { kind: WorkflowTeaserKind }) {
  if (kind === 'request') {
    return (
      <div className="workflow-teaser" aria-hidden="true">
        <svg viewBox="0 0 180 82" focusable="false">
          <rect className="teaser-frame" x="1" y="1" width="178" height="80" rx="10" />
          <path className="teaser-stroke" d="M22 24H154M22 41H128M22 58H104" />
          <path className="teaser-accent-stroke" d="M151 18V30M145 24H157" />
          <circle className="teaser-packet" r="4">
            <animateMotion path="M22 24H154" dur="2.8s" repeatCount="indefinite" />
          </circle>
        </svg>
      </div>
    );
  }

  if (kind === 'signals') {
    return (
      <div className="workflow-teaser" aria-hidden="true">
        <svg viewBox="0 0 180 82" focusable="false">
          <rect className="teaser-frame" x="1" y="1" width="178" height="80" rx="10" />
          <rect className="teaser-node-box" x="18" y="23" width="48" height="36" rx="6" />
          <path className="teaser-stroke teaser-dash" d="M68 41H157" />
          <circle className="teaser-signal teaser-signal-one" cx="100" cy="41" r="5" />
          <circle className="teaser-signal teaser-signal-two" cx="128" cy="41" r="5" />
          <circle className="teaser-signal teaser-signal-three" cx="156" cy="41" r="5" />
          <path className="teaser-accent-stroke" d="M35 44L42 37M42 37V49" />
          <circle className="teaser-packet" r="4">
            <animateMotion path="M157 41H68" dur="3.1s" repeatCount="indefinite" />
          </circle>
        </svg>
      </div>
    );
  }

  if (kind === 'agents') {
    return (
      <div className="workflow-teaser" aria-hidden="true">
        <svg viewBox="0 0 180 82" focusable="false">
          <rect className="teaser-frame" x="1" y="1" width="178" height="80" rx="10" />
          <circle className="teaser-orbit teaser-orbit-outer" cx="90" cy="41" r="29" />
          <circle className="teaser-orbit teaser-orbit-inner" cx="90" cy="41" r="20" />
          <circle className="teaser-core" cx="90" cy="41" r="12" />
          <text className="teaser-core-label" x="90" y="43">KDA</text>
          <g className="teaser-orbiting teaser-orbiting-one">
            <circle className="teaser-packet" cx="90" cy="12" r="4" />
          </g>
          <g className="teaser-orbiting teaser-orbiting-two">
            <circle className="teaser-packet teaser-packet-muted" cx="90" cy="21" r="3.5" />
          </g>
        </svg>
      </div>
    );
  }

  return (
    <div className="workflow-teaser" aria-hidden="true">
      <svg viewBox="0 0 180 82" focusable="false">
        <rect className="teaser-frame" x="1" y="1" width="178" height="80" rx="10" />
        <path className="teaser-stroke" d="M20 62H157" />
        <rect className="teaser-bar teaser-bar-one" x="28" y="45" width="18" height="17" rx="2" />
        <rect className="teaser-bar teaser-bar-two" x="58" y="34" width="18" height="28" rx="2" />
        <rect className="teaser-bar teaser-bar-three" x="88" y="20" width="18" height="42" rx="2" />
        <path className="teaser-accent-stroke teaser-result-line" d="M27 42L67 31L97 17L137 25" />
        <circle className="teaser-packet" r="4">
          <animateMotion path="M27 42L67 31L97 17L137 25" dur="3.2s" repeatCount="indefinite" />
        </circle>
        <circle className="teaser-result-check" cx="148" cy="26" r="13" />
        <path className="teaser-check-mark" d="M142 26L147 31L155 21" />
      </svg>
    </div>
  );
}

export default function Home() {
  return (
    <main>
      <nav className="nav shell" aria-label="Primary navigation">
        <a className="brand" href="#top" aria-label="Kernel Design Agents home">
          <span className="brand-mark" aria-hidden="true">KDA</span>
          <span>Kernel Design <b>Agents</b></span>
        </a>
        <div className="nav-links">
          <a href="#process">How it works</a>
          <a href="#achievements">Achievements</a>
          <a href="#faq">FAQ</a>
          <a className="nav-cta" href="https://nvlabs.github.io/kda">View Github <span aria-hidden="true">↗</span></a>
        </div>
      </nav>

      <section className="hero shell" id="top">
        <div className="hero-copy">
          <div className="eyebrow"><span /> Agentic CUDA kernel program</div>
          <h1>Community wants,<br className="hero-title-break" /> <em>KDA provides!</em></h1>
          <div className="hero-lede">
            <p>Need a CUDA kernel for your training/inference but cannot find the experts to implement?</p>
            <p>Submit your definitions and workloads to KDA-wishlist, and KDA will automatically optimize it w/ agentic workflows!</p>
            <p className="hero-affiliation">Part of NVIDIA agentic CUDA and RSI efforts</p>
          </div>
          <div className="hero-actions">
            <a className="button button-primary" href={NEW_REQUEST_URL}>
              Request a kernel <span aria-hidden="true">↗</span>
            </a>
            <a className="button button-quiet" href={ISSUES_URL}>
              Browse &amp; upvote requests
            </a>
          </div>
          <p className="hero-note">Open process. Public results. Built for real workloads.</p>
        </div>

        <div className="pipeline" aria-label="Kernel Design Agents workflow">
          <div className="pipeline-topline">
            <span>REQUEST</span>
            <span className="live"><i /> ACCEPTING</span>
          </div>
          <div className="pipeline-stage stage-input">
            <span className="stage-index">01</span>
            <div>
              <small>YOU PROVIDE</small>
              <strong>
                <a className="pipeline-link" href={FLASHINFER_TRACE_URL}>
                  FlashInfer Trace <span aria-hidden="true">↗</span>
                </a>
              </strong>
              <p>Definition · Workloads · Target</p>
            </div>
          </div>
          <div className="connector connector-research"><span className="connector-track" aria-hidden="true"><i /></span><small>research · generate · verify</small></div>
          <div className="pipeline-stage stage-process">
            <span className="stage-index">02</span>
            <div><small>WE RUN</small><strong>Kernel Design Agents</strong><p>Explore and benchmark candidates</p></div>
          </div>
          <div className="connector connector-select"><span className="connector-track" aria-hidden="true"><i /></span><small>select strongest candidate</small></div>
          <div className="pipeline-stage stage-output">
            <span className="stage-index">03</span>
            <div><small>YOU GET</small><strong>Validated kernel</strong><p>Code · Benchmarks · Reproduction</p></div>
            <span className="check" aria-hidden="true">✓</span>
          </div>
        </div>
      </section>

      <section className="definition-strip" aria-label="Program definition">
        <div className="shell definition-grid">
          <p><strong>CUDA kernels</strong> are foundational to the modern LLM ecosystem, but building high-performance implementations requires deep expertise.</p>
          <p><strong>KDA Wishlist</strong> is an experimental program where the KDA team uses agentic workflows to build the kernels the community needs.</p>
        </div>
      </section>

      <section className="section shell process-section" id="process">
        <div className="section-heading">
          <p className="section-label">HOW IT WORKS</p>
          <h2>From definitions<br />to implementations</h2>
        </div>
        <div className="section-intro">
          <p>
            Kernel Design Agents (KDA) are agentic CUDA systems that research, implement, verify, and iterate on performance-sensitive kernel tasks. Each request gives them a measurable contract—not just an idea.
          </p>
          <p>
            FlashInfer Trace is a standardized JSON format for describing a GPU kernel definition and its workload.
          </p>
          <div className="section-links">
            <a className="text-link" href={TECH_REPORT_URL}>Explore the KDA tech report <span aria-hidden="true">↗</span></a>
            <a className="text-link" href={FLASHINFER_TRACE_URL}>Explore the FlashInfer-Trace spec <span aria-hidden="true">↗</span></a>
          </div>
        </div>
      </section>

      <section className="workflow-shell shell">
        <div className="workflow-head">
          <span>PUBLIC REQUEST LIFECYCLE</span>
          <span>GITHUB-MANAGED</span>
        </div>
        <div className="workflow-grid">
          {workflow.map(({ number, title, copy, teaser }) => (
            <article className="workflow-step" key={number}>
              <div className="step-number">{number}</div>
              <WorkflowTeaser kind={teaser} />
              <h3>{title}</h3>
              <p>{copy}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="section shell achievements-section" id="achievements">
        <div className="achievements-intro">
          <div>
            <p className="section-label">SELECTED ACHIEVEMENTS</p>
            <h2 className="achievements-title">Built by agents.<br /><span>Merged upstream.</span></h2>
          </div>
          <div className="achievements-copy">
            <p>
              KDA has already turned optimization targets into production contributions across LLM serving, image generation, and video workloads.
            </p>
          </div>
        </div>

        <div className="achievements-grid" aria-label="Selected KDA kernel achievements">
          {achievements.map(({ metric, metricLabel, title, benchmark, id, project, status, linkLabel, href }, index) => (
            <article className="achievement-card" key={id}>
              <div className="achievement-card-top">
                <span>{String(index + 1).padStart(2, '0')} / {project}</span>
                <span className="achievement-status"><i /> {status}</span>
              </div>
              <div className="achievement-metric">
                <strong>{metric}</strong>
                <span>{metricLabel}</span>
              </div>
              <div className="achievement-body">
                <h3>{title}</h3>
                <p>{benchmark}</p>
              </div>
              <div className="achievement-links">
                <a className="achievement-link" href={href}>
                  {linkLabel} <span aria-hidden="true">↗</span>
                </a>
                {id === 'kda-0.1-flashinfer-contest' && (
                  <a className="achievement-link" href={KDA_01_CONTEST_SOLUTIONS_URL}>
                    Contest Solutions <span aria-hidden="true">↗</span>
                  </a>
                )}
              </div>
            </article>
          ))}
        </div>

        <div className="more-improvements" id="more-improvements">
          <p className="more-improvements-label">More improvements</p>
          <ul className="improvements-list" aria-label="More KDA improvements">
            {moreImprovements.map(({ result, operator, repoLabel, href }) => (
              <li className="improvement-line" key={repoLabel}>
                <strong>{result}</strong> · {operator} · <a href={href}>{repoLabel} <span aria-hidden="true">↗</span></a>
              </li>
            ))}
          </ul>
        </div>

        <p className="achievements-note">
          Results vary by hardware, model, and workload. See each linked pull request or repository for validation and reproduction details.
        </p>
      </section>

      <section className="section shell faq-section" id="faq">
        <div className="section-heading">
          <p className="section-label">GOOD TO KNOW</p>
          <h2>Before you<br />request a kernel</h2>
        </div>
        <div className="faq-list">
          <details>
            <summary>How do upvotes work?<span aria-hidden="true">+</span></summary>
            <p>Open the public wishlist, find a request you care about, and add a thumbs-up reaction to the top-level GitHub issue. Comments are best used for new workload evidence or implementation context.</p>
          </details>
          <details>
            <summary>Will my request and results be public?<span aria-hidden="true">+</span></summary>
            <p>
              Yes—unless we agree otherwise in advance. For accepted requests, KDA may publish the submitted definitions, generated implementations, benchmarks, profiling data, and design notes. We credit submitters for their definitions and workloads. We are also collecting representative kernels and workloads to guide the development of the next generation of KDA. If your business requires private kernels, email <a href="mailto:ligengz@nvidia.com">ligengz@nvidia.com</a> to discuss a collaboration through NVIDIA Kernel Factory.
            </p>
          </details>
          <details>
            <summary>How many kernels will KDA deliver?<span aria-hidden="true">+</span></summary>
            <p>
              KDA typically delivers one to three of the most-requested kernels each week, depending on the team’s workloads. Sihao Liu and Ligeng Zhu are currently self-paying the GPU and tokens, so capacity is limited. If you have a bunch of kernels to optimize, email <a href="mailto:ligengz@nvidia.com">ligengz@nvidia.com</a> to setup an official collaboration through NVIDIA Kernel Factory.
            </p>
          </details>
          <details>
            <summary>Why might a request be returned or declined?<span aria-hidden="true">+</span></summary>
            <p>Requests may need revision when they cannot be reproduced, lack a reliable correctness reference, target unsupported hardware, are too vague for automated evaluation, or have unclear licensing.</p>
          </details>
        </div>
      </section>

      <section className="final-cta shell">
        <div>
          <p className="section-label light">YOUR BOTTLENECK, NEXT</p>
          <h2>What should<br />KDA build?</h2>
        </div>
        <div className="final-actions">
          <p>Bring the definition and real workloads. We’ll bring the research loop.</p>
          <a className="button button-acid" href={NEW_REQUEST_URL}>Submit a kernel request <span aria-hidden="true">↗</span></a>
          <a className="final-secondary" href={ISSUES_URL}>Or browse the wishlist and upvote <span aria-hidden="true">→</span></a>
        </div>
      </section>

      <footer className="footer shell">
        <a className="brand" href="#top" aria-label="Kernel Design Agents home">
          <span className="brand-mark" aria-hidden="true">KDA</span>
          <span>Kernel Design <b>Agents</b></span>
        </a>
        <p>An agentic-driven CUDA project</p>
        <div>
          <a href={ISSUES_URL}>GitHub</a>
          <a href="https://github.com/humanfia/humanize2">Humanize</a>
        </div>
      </footer>
    </main>
  );
}
