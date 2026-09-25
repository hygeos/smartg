"""GPU-free tests of the inverse CDF of smartg.cdf."""

import numpy as np
import pytest

from smartg.cdf import icdf


@pytest.mark.parametrize(
    ("pdf", "counts"),
    [
        # a zero probability after the first one
        ([1.0, 0.0, 2.0], [10, 0, 20]),
        # a small first probability
        ([1e-3, 1.0, 1.0], [10, 10000, 10000]),
    ],
)
def test_icdf_samples_the_smallest_probability(
    pdf: list[float], counts: list[int]
) -> None:
    """The smallest non-zero probability gets 10 samples, zero none."""
    samples = icdf(pdf)
    np.testing.assert_array_equal(np.bincount(samples, minlength=3), counts)


@pytest.mark.parametrize(
    "pdf", [[0.0, 0.0], [1.0, np.nan], [1.0, -0.5, 1.0], [1e-12, 1.0]]
)
def test_icdf_refuses_a_pdf_it_cannot_size(pdf: list[float]) -> None:
    """A pdf without a usable smallest probability raises."""
    with pytest.raises(ValueError, match="pdf"):
        icdf(pdf)


def test_icdf_given_n_is_unchanged() -> None:
    """A given n keeps the sampling of the mid-points."""
    np.testing.assert_array_equal(icdf([1.0, 3.0], n=4), [0, 1, 1, 1])
