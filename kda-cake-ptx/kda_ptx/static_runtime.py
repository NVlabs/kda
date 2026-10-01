# Copyright 2026 KDA(Kernel Design Agents) Team
# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Load retained PTX kernels through generated CUDA host shims.

The retained PTX is ``.version 9.2`` / ``.target sm_103a``. Drivers older than
CUDA 13.2 cannot JIT it, so each file is assembled once with ptxas >= 13.2 and
the cubin is embedded instead. The cubins are cached in ``KDA_PTX_CACHE_DIR``
(default ``~/.cache/kda-ptx``); ``KDA_PTXAS`` selects a specific ptxas.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from functools import cache
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
SHIMS = HERE / "shims"
PTX_ARCH = "sm_103a"
PTXAS_MIN_VERSION = (13, 2)


def _cuda_include() -> str:
    import triton

    root = Path(triton.__file__).resolve().parent
    include = root / "backends/nvidia/include"
    if not (include / "cuda.h").is_file():
        raise RuntimeError("CUDA driver headers are unavailable")
    return str(include)


def _ptxas_version(path: str) -> tuple[int, int]:
    text = subprocess.run(
        [path, "--version"], capture_output=True, text=True, check=True
    ).stdout
    match = re.search(r"release (\d+)\.(\d+)", text)
    return (int(match[1]), int(match[2])) if match else (0, 0)


@cache
def _ptxas() -> tuple[str, tuple[int, int]]:
    """Find a ptxas new enough for PTX ISA 9.2.

    Order: ``KDA_PTXAS``, the nvidia-cuda-nvcc wheel, ``CUDA_HOME``, ``PATH``.
    """
    candidates = [os.environ.get("KDA_PTXAS")]
    candidates += [str(Path(p) / "nvidia/cu13/bin/ptxas") for p in sys.path if p]
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        candidates.append(str(Path(cuda_home) / "bin/ptxas"))
    candidates += [shutil.which("ptxas"), "/usr/local/cuda/bin/ptxas"]
    for candidate in candidates:
        if not candidate or not os.access(candidate, os.X_OK):
            continue
        try:
            version = _ptxas_version(candidate)
        except (OSError, subprocess.CalledProcessError):
            continue
        if version >= PTXAS_MIN_VERSION:
            return candidate, version
    raise RuntimeError(
        "the retained PTX needs ptxas >= 13.2 (install nvidia-cuda-nvcc>=13.2 "
        "or set KDA_PTXAS)"
    )


def _cubin(name: str) -> bytes:
    """Assemble one retained PTX file for sm_103a, cached by content."""

    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (10, 3):
        raise RuntimeError(
            f"the retained PTX targets {PTX_ARCH} (B300); this GPU is "
            f"sm_{major}{minor}"
        )
    source = SHIMS / f"{name}.ptx"
    ptxas, version = _ptxas()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    cache_dir = Path(
        os.environ.get("KDA_PTX_CACHE_DIR", Path.home() / ".cache" / "kda-ptx")
    )
    cubin = cache_dir / (
        f"{name}-{digest}-ptxas{version[0]}.{version[1]}.{PTX_ARCH}.cubin"
    )
    if not cubin.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        partial = cubin.with_name(f"{cubin.name}.{os.getpid()}.tmp")
        subprocess.run(
            [ptxas, f"-arch={PTX_ARCH}", "-o", str(partial), str(source)],
            check=True,
        )
        partial.replace(cubin)
    return cubin.read_bytes()


@cache
def _load(name: str):
    from tvm_ffi import cpp

    metadata = json.loads((SHIMS / f"{name}.json").read_text())
    source = (SHIMS / f"{name}.cc").read_text()
    module = cpp.load_inline(
        f"best_ptx_{metadata['module_ident']}",
        cpp_sources=source,
        embed_cubin={metadata["module_ident"]: _cubin(name)},
        extra_include_paths=[_cuda_include()],
        extra_ldflags=["-lcuda"],
    )
    keys = [key for kind, key in metadata["arg_plan"] if kind != "grid"]
    tma_keys = [key for kind, key in metadata["arg_plan"] if kind == "tma_buffer"]
    return module[metadata["entry"]], keys, tma_keys


class StaticPTXModule:
    """Kernel-module surface used by the retained host scheduler.

    Each module owns a device table with one TMA descriptor per ``tma_buffer``
    argument. ``bind`` encodes and uploads it for a set of bindings (outside
    CUDA Graph capture); ``launch`` re-uploads only when a TMA tensor changed,
    so a prepared launch does no host-to-device work.
    """

    def __init__(self, name: str):
        self.name = name
        self._tma_key = None
        self._tma_table = None

    def bind(self, bindings):
        """Upload the TMA descriptor table for these bindings if it changed."""

        import tvm_ffi

        entry, keys, tma_keys = _load(self.name)
        tensors = [bindings[key] for key in tma_keys]
        key = tuple(
            (t.data_ptr(), tuple(t.shape), t.stride(), t.dtype) for t in tensors
        )
        if key == self._tma_key:
            return
        table = torch.empty(
            len(tma_keys) * 128, dtype=torch.uint8, device=tensors[0].device
        )
        upload = {**bindings, "tma_table": table, "tma_upload": 1}
        with tvm_ffi.use_torch_stream():
            # An upload call launches nothing; the grid only has to pass the
            # shims' checks (cluster-2 kernels need an even grid).
            entry(*[upload[k] for k in keys], 2, 1, 1)
        self._tma_key, self._tma_table = key, table

    def launch(self, *, grid, **bindings):
        import tvm_ffi

        entry, keys, _ = _load(self.name)
        self.bind(bindings)
        bindings["tma_table"] = self._tma_table
        bindings["tma_upload"] = 0
        if self.name == "varlen_mixed_h64" and grid[0] % 2:
            bindings["cta_table"] = torch.cat(
                (
                    bindings["cta_table"],
                    torch.zeros(
                        3, dtype=torch.int32, device=bindings["cta_table"].device
                    ),
                )
            )
            grid = (grid[0] + 1, grid[1], grid[2])
        missing = set(keys) - set(bindings)
        if missing:
            raise ValueError(f"missing PTX launch bindings: {sorted(missing)}")
        args = [bindings[key] for key in keys]
        with tvm_ffi.use_torch_stream():
            entry(*args, *grid)


def compiled_stream_m64_fixed_final(*, split_major=False, **_):
    return StaticPTXModule("fixed_h64") if split_major else None


def compiled_stream_m64_fixed():
    return None


def compiled_stream_m128_fixed_h96_final(**_):
    return StaticPTXModule("fixed_h96")


def compiled_stream_m128_cluster8(**_):
    return None


def compiled_stream_m128_cluster2(
    *, full_chunks=False, store_final_state=True, handoff=False, **_
):
    if not full_chunks and store_final_state and handoff:
        return StaticPTXModule("varlen_mixed_h64")
    return None


def compiled_stream_m128(
    *, full_chunks=False, store_final_state=True, handoff=False, **_
):
    if full_chunks and store_final_state:
        return StaticPTXModule("varlen_uniform_h64")
    if not full_chunks and store_final_state:
        return StaticPTXModule("varlen_mixed_h64")
    return None


__all__ = [
    "StaticPTXModule",
    "compiled_stream_m64_fixed",
    "compiled_stream_m64_fixed_final",
    "compiled_stream_m128",
    "compiled_stream_m128_cluster2",
    "compiled_stream_m128_cluster8",
    "compiled_stream_m128_fixed_h96_final",
]
