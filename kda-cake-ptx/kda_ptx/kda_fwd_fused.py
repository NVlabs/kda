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

"""Independent static-PTX KDA-forward entry point."""

from __future__ import annotations

import torch

from .scheduler import prepare_fwd_pieces


class KDAForwardLaunch:
    def __init__(
        self,
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        scale,
        *,
        out,
        lower_bound=-5.0,
        initial_state=None,
        final_state=None,
        cu_seqlens=None,
        allow_approximate_split=False,
        use_expected_norm=False,
    ):
        self.out = out
        self.final_state = final_state
        self._fused = prepare_fwd_pieces(
            q,
            k,
            v,
            g,
            beta,
            scale,
            out,
            A_log,
            dt_bias,
            lower_bound,
            initial_state=initial_state,
            final_state=final_state,
            cu_seqlens=cu_seqlens,
            allow_approximate_split=allow_approximate_split,
            use_expected_norm=use_expected_norm,
        )

    @property
    def schedule(self):
        return self._fused.schedule

    @property
    def state_input_mode(self):
        return self._fused.state_input_mode

    @property
    def normalization_mode(self):
        return self._fused.normalization_mode

    def launch(self, g=None):
        self._fused.launch(g=g)
        return self.out, self.final_state


def prepare_kda_forward(
    q,
    k,
    v,
    g,
    beta,
    A_log,
    dt_bias,
    scale,
    *,
    out=None,
    lower_bound=-5.0,
    initial_state=None,
    final_state=None,
    cu_seqlens=None,
    allow_approximate_split=False,
    use_expected_norm=False,
):
    if dt_bias.ndim == 1:
        dt_bias = dt_bias.view(q.shape[2], q.shape[3])
    if out is None:
        out = torch.empty_like(q)
    return KDAForwardLaunch(
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        scale,
        out=out,
        lower_bound=lower_bound,
        initial_state=initial_state,
        final_state=final_state,
        cu_seqlens=cu_seqlens,
        allow_approximate_split=allow_approximate_split,
        use_expected_norm=use_expected_norm,
    )


def kda_forward(*args, **kwargs):
    return prepare_kda_forward(*args, **kwargs).launch()


__all__ = ["KDAForwardLaunch", "kda_forward", "prepare_kda_forward"]
