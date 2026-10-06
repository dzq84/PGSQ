from __future__ import annotations

import pytest
import torch


@pytest.mark.gpu
def test_cuda_tensor_roundtrip() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    value = torch.ones(1, device="cuda")
    assert value.cpu().item() == 1
