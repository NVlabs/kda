"""Prepared-launch API for the direct CuTe DSL KDA forward kernel."""

from __future__ import annotations

import torch

from .forward import fwd


class KDAForwardLaunch:
    """Preallocated exact KDA forward launch with final-state export."""

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
        initial_state,
        final_state,
        cu_seqlens=None,
        lower_bound=-5.0,
    ):
        self.q = q
        self.k = k
        self.v = v
        self.g = g
        self.beta = beta
        self.A_log = A_log
        self.dt_bias = dt_bias
        self.scale = float(scale)
        self.out = out
        self.initial_state = initial_state
        self.final_state = final_state
        self.cu_seqlens = cu_seqlens
        self.lower_bound = float(lower_bound)
        self.state_input_mode = "native_fp32"
        self.normalization_mode = "exact_norm"
        self.schedule = "cute_direct_m128_persistent_exact"

    def launch(self, g=None):
        """Launch once, optionally binding a same-shaped gate tensor."""
        return fwd(
            self.q,
            self.k,
            self.v,
            self.g if g is None else g,
            self.beta,
            self.scale,
            self.out,
            self.A_log,
            self.dt_bias,
            self.lower_bound,
            initial_state=self.initial_state,
            final_state=self.final_state,
            cu_seqlens=self.cu_seqlens,
            allow_approximate_split=False,
        )


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
    initial_state,
    final_state,
    cu_seqlens=None,
    lower_bound=-5.0,
):
    """Create an exact-normalization launch with approximate splits disabled."""
    if out is None:
        out = torch.empty_like(v)
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
        initial_state=initial_state,
        final_state=final_state,
        cu_seqlens=cu_seqlens,
        lower_bound=lower_bound,
    )


def kda_forward(*args, **kwargs):
    """Run one exact direct-CuTe forward call."""
    return prepare_kda_forward(*args, **kwargs).launch()


__all__ = ["KDAForwardLaunch", "kda_forward", "prepare_kda_forward"]
