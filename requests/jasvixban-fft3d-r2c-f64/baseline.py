# SPDX-License-Identifier: Apache-2.0
"""Best-known baseline: cuFFT, reached through torch.fft (see README for
source/version; torch.fft.rfftn dispatches to cuFFT D2Z for float64 CUDA
tensors).
"""
import torch


def run(input):
    return torch.view_as_real(torch.fft.rfftn(input, norm="backward")).contiguous()
