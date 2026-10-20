"""GPU-free tests of the interpolation helpers of smartg.interp."""

import numpy as np
import pytest

from smartg.interp import _interp2_stack, interp2


@pytest.mark.parametrize("decreasing", [False, True])
def test_interp2_stack_as_interp2(decreasing: bool) -> None:
    """Each array of the stack is sampled as interp2 samples it."""
    rng = np.random.default_rng(1)
    x = np.array([1.0, 10.0, 100.0, 1000.0])
    y = np.array([180.0, 220.0, 260.0, 300.0, 340.0])
    if decreasing:
        x = x[::-1]
    arrays = [rng.random((4, 5)) for _ in range(6)]
    xi = np.array([[0.5, 50.0, 999.0], [1200.0, 3.0, 400.0]])
    yi = np.array([[170.0, 250.0, 300.0], [350.0, 181.0, 299.9]])
    stack = _interp2_stack(x, y, arrays, xi, yi)
    assert stack.shape == (6, 2, 3)
    for out, a in zip(stack, arrays, strict=True):
        assert out.tobytes() == interp2(x, y, a, xi, yi).tobytes()
