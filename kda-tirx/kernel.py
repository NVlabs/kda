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

"""kda_forward two-phase wrapper for jinhongyii's TIRx KDA forward gist
(gist 4c50fafe314b63914974619faa3f9985, kda_fwd_varlen.py).

The gist's own protocol is ``setup(data, B, T, H) -> run``: setup compiles, encodes
tensor maps, warms up, synchronizes and records a private CUDA graph; run replays it.
That maps onto the judge's ``prepare(*inputs) -> launch`` form directly. The only edit
to the gist is the hard-coded ``sm_100a``, which becomes the device's architecture.
"""

import types
from pathlib import Path

import torch

_SRC = Path(__file__).with_name("kda_fwd_varlen.py").read_text()
_major, _minor = torch.cuda.get_device_capability()
_module = types.ModuleType("kda_fwd_varlen_gist")
exec(compile(_SRC.replace("sm_100a", f"sm_{_major}{_minor}a"), "kda_fwd_varlen_gist", "exec"), _module.__dict__)


def prepare(q, k, v, g, beta, A_log, dt_bias, scale, initial_state, cu_seqlens=None):
    out = torch.empty_like(v)
    final_state = torch.empty_like(initial_state)
    data = {
        "q": q, "k": k, "v": v, "g": g, "beta": beta,
        "A_log": A_log, "dt_bias": dt_bias, "scale": float(scale),
        "initial_state": initial_state, "output": out, "final_state": final_state,
        "cu_seqlens": cu_seqlens,
    }
    gist_run = _module.setup(data, 1, q.shape[1], q.shape[2])

    def launch():
        gist_run()
        return out, final_state

    return launch


def run(q, k, v, g, beta, A_log, dt_bias, scale, initial_state, cu_seqlens=None):
    return prepare(q, k, v, g, beta, A_log, dt_bias, scale, initial_state, cu_seqlens)()
