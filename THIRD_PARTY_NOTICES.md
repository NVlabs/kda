# Third-Party Notices

This file identifies third-party material distributed with Kernel Design
Agents. Third-party components are not covered by the first-party mixed license
in [LICENSE](LICENSE). Copyright notices, file-level SPDX identifiers, and
bundle provenance records remain applicable.

Verbatim copies of the upstream license files referenced below are provided in
[`third_party_licenses/`](third_party_licenses/README.md).

## Included Git Submodules

### KernelWiki

- Path: `skills/KernelWiki`
- Source: <https://github.com/mit-han-lab/KernelWiki.git>
- Upstream base revision: `76d27b56f804e7e7295d4c570e1e5d7eef4b0a75`
- Distributed revision: the gitlink recorded at `skills/KernelWiki`
- License for original KernelWiki material: MIT
- License text: [`third_party_licenses/MIT-hanlab.txt`](third_party_licenses/MIT-hanlab.txt)

KernelWiki includes verbatim, extracted, and derived artifact bundles. The
license of an artifact is determined by its upstream project and any
file-specific notice, not by KernelWiki's MIT license. The component inventory
below covers all 89 bundles in the distributed checkout.

### ncu-report-skill

- Path: `skills/ncu-report-skill`
- Source: <https://github.com/mit-han-lab/ncu-report-skill.git>
- Pinned revision: `d1887948c7d53690cfe6605f59c1329b8a1c6bb5`
- License: MIT
- Copyright: Copyright (c) 2026 hanlab
- License text: [`third_party_licenses/MIT-hanlab.txt`](third_party_licenses/MIT-hanlab.txt)

## KernelWiki Artifact Components

| Upstream component | Bundles | Distributed terms | License text |
|---|---:|---|---|
| [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass) | 20 | BSD-3-Clause for distributed files; restricted CuTe DSL EULA material is omitted as described below | [`BSD-3-Clause-CUTLASS.txt`](third_party_licenses/BSD-3-Clause-CUTLASS.txt) |
| [FlashInfer](https://github.com/flashinfer-ai/flashinfer) | 9 | Apache-2.0 at repository level; included NVIDIA-origin files retain BSD-3-Clause notices | [`Apache-2.0-FlashInfer.txt`](third_party_licenses/Apache-2.0-FlashInfer.txt), [`BSD-3-Clause-CUTLASS.txt`](third_party_licenses/BSD-3-Clause-CUTLASS.txt) |
| [PyTorch](https://github.com/pytorch/pytorch) | 5 | PyTorch BSD-style license and copyright notices | [`BSD-3-Clause-PyTorch.txt`](third_party_licenses/BSD-3-Clause-PyTorch.txt) |
| [SGLang](https://github.com/sgl-project/sglang) | 10 | Apache-2.0 | [`Apache-2.0-SGLang.txt`](third_party_licenses/Apache-2.0-SGLang.txt) |
| [vLLM](https://github.com/vllm-project/vllm) | 20 | Apache-2.0 | [`Apache-2.0-vLLM.txt`](third_party_licenses/Apache-2.0-vLLM.txt) |
| [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) | 2 | MIT | [`MIT-DeepGEMM.txt`](third_party_licenses/MIT-DeepGEMM.txt) |
| KernelWiki-derived artifacts | 12 | KernelWiki MIT license | [`MIT-hanlab.txt`](third_party_licenses/MIT-hanlab.txt) |
| Source-attributed blog and contest excerpts | 11 | Source-specific terms recorded by each bundle's `PROVENANCE.yaml`; no blanket relicensing is asserted here | Source links listed below |

The license copies were taken from these upstream revisions:

- CUTLASS `2252254c`: <https://github.com/NVIDIA/cutlass/blob/2252254c/LICENSE.txt>
- FlashInfer `18804cd5`: <https://github.com/flashinfer-ai/flashinfer/blob/18804cd5/LICENSE>
- PyTorch `3c40486f`: <https://github.com/pytorch/pytorch/blob/3c40486f/LICENSE>
- SGLang `260abe1f`: <https://github.com/sgl-project/sglang/blob/260abe1f/LICENSE>
- vLLM `074854b2`: <https://github.com/vllm-project/vllm/blob/074854b2/LICENSE>
- DeepGEMM `891d57b4db1071624b5c8fa0d1e51cb317fa709f`: <https://github.com/deepseek-ai/DeepGEMM/blob/891d57b4db1071624b5c8fa0d1e51cb317fa709f/LICENSE>
- KernelWiki `b6b4301f15e8ce6955a56776690643ce5db369e6`: <https://github.com/mit-han-lab/KernelWiki/blob/b6b4301f15e8ce6955a56776690643ce5db369e6/LICENSE>
- ncu-report-skill `74a12918e9f64d78036f14da5f8765e435b949a4`: <https://github.com/mit-han-lab/ncu-report-skill/blob/74a12918e9f64d78036f14da5f8765e435b949a4/LICENSE>

## Source-Attributed Excerpts

The following sources account for the 11 blog and contest-excerpt bundles.
Each bundle preserves its source URL and attribution details in
`PROVENANCE.yaml`:

- [Colfax Research CUTLASS Blackwell tutorial](https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tmem-for-nvidia-blackwell-gpus/)
- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)
- [FlashAttention-4](https://tridao.me/blog/2026/flash4/)
- [FlashMLA](https://github.com/deepseek-ai/FlashMLA)
- [GatedDeltaNet](https://github.com/NVlabs/GatedDeltaNet)
- [Simon Veitner's NVFP4 GEMV article](https://veitner.bearblog.dev/nvfp4-gemv/)
- [tcgen05 tutorial](https://gau-nernst.github.io/tcgen05/)
- [Hugging Face APSys Blackwell NVFP4 comparison](https://huggingface.co/blog/apsys/blackwell-nvfp4-comparison)
- [Yue Zhang's Blackwell NVFP4 hackathon article](https://yue-zhang-2025.github.io/2025/12/02/blackwell-nvfp4-kernel-hackathon-journey.html)

## Restricted CUTLASS CuTe DSL Material

The upstream CUTLASS PR bundles for PRs 2750, 3021, 3091, 3106, and 3130
contained 19 files carrying NVIDIA's proprietary SPDX identifier. Those source
snapshots are not distributed.

To avoid retaining restricted excerpts without their header, three additional
mixed or unmarked `diff.patch` files were also omitted. In total, 22 artifact
payload files were removed while the source PR pages and provenance metadata
were retained. Each affected `PROVENANCE.yaml` lists the omitted paths and the
reason for omission. BSD-3-Clause files in the PR-3091 and PR-3130 bundles
remain distributed.

The upstream EULA referenced by those files is available at
<https://docs.nvidia.com/cutlass/media/docs/pythonDSL/license.html>. KernelWiki's
validator rejects any artifact payload that reintroduces the restricted SPDX
identifier.

## Preservation Requirements

- Keep each artifact's existing copyright and SPDX notices intact.
- Keep each bundle's `PROVENANCE.yaml` with the distributed artifact files.
- Do not apply the KernelWiki or Kernel Design Agents license to third-party
  artifact content.
- Re-run `python3 scripts/validate.py` in `skills/KernelWiki` before release.
