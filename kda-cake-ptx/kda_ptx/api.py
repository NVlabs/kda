"""Public API for the static PTX KDA forward implementation."""

from __future__ import annotations

import torch

from .kda_fwd_fused import prepare_kda_forward


def prepare(q, k, v, g, beta, A_log, dt_bias, scale, initial_state, cu_seqlens=None):
    """Prepare one exact launch returning output and FP32 final state."""

    out = torch.empty_like(q)
    final_state = torch.empty_like(initial_state)
    launch = prepare_kda_forward(
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        scale,
        out=out,
        initial_state=initial_state,
        final_state=final_state,
        cu_seqlens=cu_seqlens,
        allow_approximate_split=False,
        use_expected_norm=False,
    )

    def invoke(bound_g=None):
        return launch.launch(g=bound_g)

    return invoke


def run(*args):
    """Run one exact KDA forward call through static PTX kernels."""

    return prepare(*args)()


def fwd(
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
    initial_state=None,
    final_state=None,
    cu_seqlens=None,
    allow_approximate_split=False,
):
    """Run the preallocated interface used by the benchmark harness."""

    launch = prepare_kda_forward(
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
        use_expected_norm=False,
    )
    return launch.launch()


__all__ = ["fwd", "prepare", "prepare_kda_forward", "run"]
