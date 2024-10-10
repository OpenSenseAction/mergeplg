import numpy as np
import xarray as xr

import mergeplg as mrg

# from .test_radolan_adjust import get_test_data


def test_rounding_down():
    a = np.array([0.0999999, 0.11111, 0.19999, 1.99999])

    np.testing.assert_almost_equal(
        mrg.radolan.processing.round_down(a, decimal=0), np.array([0.0, 0.0, 0.0, 1])
    )

    np.testing.assert_almost_equal(
        mrg.radolan.processing.round_down(a, decimal=1), np.array([0.0, 0.1, 0.1, 1.9])
    )

    np.testing.assert_almost_equal(
        mrg.radolan.processing.round_down(a, decimal=2),
        np.array([0.09, 0.11, 0.19, 1.99]),
    )

    # also test passing xr.DataArray
    np.testing.assert_almost_equal(
        mrg.radolan.processing.round_down(xr.DataArray(a), decimal=2),
        np.array([0.09, 0.11, 0.19, 1.99]),
    )
