# SPDX-License-Identifier: Apache-2.0
"""Best-known baseline: cuFFT, reached through torch.fft (see README for
source/version; torch.fft.irfftn dispatches to cuFFT Z2D for float64 CUDA
tensors).

norm="forward" leaves the inverse unscaled, matching cuFFT Z2D (torch's
default norm="backward" would divide by N, which the contract forbids).
"""
import torch


def run(input):
    z = torch.view_as_complex(input.contiguous())
    nz, ny, nf = z.shape
    return torch.fft.irfftn(z, s=(nz, ny, 2 * (nf - 1)), norm="forward")
