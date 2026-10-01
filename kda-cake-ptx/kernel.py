"""kda_forward task entry point: the static PTX implementation in kda_ptx/.

``prepare`` plans once (host scheduling, device tables) and returns a launch that
can be captured in a CUDA graph; ``run`` plans and launches in one call.
"""

from kda_ptx import prepare, run  # noqa: F401
