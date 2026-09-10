# SPDX-License-Identifier: Apache-2.0
"""FlashInfer RMSNorm comparison implementation; see README.md for source/version."""

from flashinfer.norm import rmsnorm


def run(input, weight, eps):
    return rmsnorm(input, weight, eps=eps)
