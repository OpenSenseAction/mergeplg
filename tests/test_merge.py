"""
Created on Mon Jul 29 10:03:34 2024

@author: Erlend Øydvin
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from mergeplg import merge

ds_cmls = xr.Dataset(
    data_vars={
        "R": (("cml_id", "time"), np.reshape(np.arange(1, 13), (3, 4))),
    },
    coords={
        "cml_id": ("cml_id", ["cml1", "cml2", "cml3"]),
        "time": ("time", np.arange(0, 4)),
        "site_0_x": ("cml_id", [-1, 0, 0]),
        "site_0_y": ("cml_id", [-1, -1, 1]),
        "site_1_x": ("cml_id", [1, 2, 2]),
        "site_1_y": ("cml_id", [1, 1, 3]),
        "x": ("cml_id", [0, 1, 1]),
        "y": ("cml_id", [0, 0, 2]),
        "length": ("cml_id", [2 * np.sqrt(2), 2 * np.sqrt(2), 2 * np.sqrt(2)]),
    },
)

ds_rad = xr.Dataset(
    data_vars={"R": (("y", "x", "time"), np.ones([4, 4, 4]))},
    coords={
        "x": ("x", [-1, 0, 1, 2]),
        "y": ("y", [-1, 0, 1, 2]),
        "time": ("time", [0, 1, 2, 3]),
        "xs": (("y", "x"), np.meshgrid([-1, 0, 1, 2], [-1, 0, 1, 2])[0]),
        "ys": (("y", "x"), np.meshgrid([-1, 0, 1, 2], [-1, 0, 1, 2])[1]),
    },
)


def test_calculate_cml_geometry():
    # Test that the CML geometry is correctly esimtated
    y, x = merge.calculate_cml_geometry(
        ds_cmls.isel(cml_id=[1]),
        disc=2,  # divides the line into two intervals, 3 points
    )[0]
    assert (y == np.array([-1, 0, 1])).all()
    assert (x == np.array([0, 1, 2])).all()


def test_block_points_to_lengths():
    # Check that the length matrix is correctly estimated
    line = merge.block_points_to_lengths(
        merge.calculate_cml_geometry(ds_cmls.isel(cml_id=[0, 1]), disc=2)
    )
    l0l0 = line[0, 0]  # Lengths from link0 to link0
    l0l1 = line[0, 1]  # Lengths from link0 to link0
    l1l0 = line[1, 0]  # Lengths from link0 to link0
    l1l1 = line[1, 1]  # Lengths from link0 to link0

    # Length matrix from line1 to lin1
    assert (
        l0l0
        == np.array(
            [
                [0, np.sqrt(2), 2 * np.sqrt(2)],
                [np.sqrt(2), 0, np.sqrt(2)],
                [2 * np.sqrt(2), np.sqrt(2), 0],
            ]
        )
    ).all()

    # Length matrix from line1 to lin2
    assert (
        l0l1
        == np.array(
            [
                [1, 1, np.sqrt(2**2 + 1)],
                [np.sqrt(2**2 + 1), 1, 1],
                [np.sqrt(3**2 + 2**2), np.sqrt(2**2 + 1), 1],
            ]
        )
    ).all()

    # Length matrix from line2 to lin1
    assert (
        l1l0
        == np.array(
            [
                [1, np.sqrt(2**2 + 1), np.sqrt(3**2 + 2**2)],
                [1, 1, np.sqrt(2**2 + 1)],
                [np.sqrt(1 + 2**2), 1, 1],
            ]
        )
    ).all()

    # Length matrix from line2 to lin2
    assert (
        l1l1
        == np.array(
            [
                [0, np.sqrt(2), 2 * np.sqrt(2)],
                [np.sqrt(2), 0, np.sqrt(2)],
                [2 * np.sqrt(2), np.sqrt(2), 0],
            ]
        )
    ).all()


def test_merge_additive_idw():
    # Calculate CML-radar difference
    ds_cmls["R_diff"] = xr.full_like(ds_cmls.R, 0)
    for cml_id in ds_cmls.cml_id:
        rad_r = ds_rad.sel(
            x=ds_cmls.sel(cml_id=cml_id).x,
            y=ds_cmls.sel(cml_id=cml_id).y,
        ).R
        cml_r = ds_cmls.sel(cml_id=cml_id).R
        ds_cmls["R_diff"].loc[{"cml_id": cml_id}] = cml_r - rad_r

    additive_idw = merge.merge_additive_idw(
        ds_cmls.R_diff.isel(time=0), ds_rad.R.isel(time=0), where_rad=True, min_obs=3
    )

    print(additive_idw)

    # Check that radar is conditioned to point observations
    for cml_id in ds_cmls.cml_id:
        merge_r = additive_idw.sel(
            x=ds_cmls.sel(cml_id=cml_id).x,
            y=ds_cmls.sel(cml_id=cml_id).y,
        ).data
        cml_r = ds_cmls.sel(cml_id=cml_id).isel(time=0).R.data
        assert cml_r == merge_r


def test_merge_additive_blockkriging():
    # Calculate CML-radar difference
    ds_cmls["R_diff"] = xr.full_like(ds_cmls.R, 0)
    for cml_id in ds_cmls.cml_id:
        rad_r = ds_rad.sel(
            x=ds_cmls.sel(cml_id=cml_id).x,
            y=ds_cmls.sel(cml_id=cml_id).y,
        ).R
        cml_r = ds_cmls.sel(cml_id=cml_id).R
        ds_cmls["R_diff"].loc[{"cml_id": cml_id}] = cml_r - rad_r

    # Calculate CML geometry
    x0 = merge.calculate_cml_geometry(ds_cmls)

    # Define variogram (exponential)
    def variogram(h):  # Exponential variogram
        return 0 + (1 - 0) * (1 - np.exp(-h * 3 / 1))

    # do additive blockkriging
    additive_blockkriging = merge.merge_additive_blockkriging(
        ds_cmls.R_diff.isel(time=0),
        ds_rad.R.isel(time=0),
        x0,
        variogram,
        where_rad=True,
        min_obs=3,
    )

    # Is not strightforward to check the line integrals as that would be an
    # approximation. We therefore check the full matrix against a copy for now.
    data_check = np.array(
        [
            [1.44257989, 4.60679755, 5.20351533, 5.24607484],
            [4.20176434, 0.44431991, 4.85025407, 5.28417687],
            [5.25647569, 7.53645848, 2.51821776, 5.64689963],
            [5.32243757, 6.10865941, 9.55481817, 6.18926484],
        ]
    )

    np.testing.assert_almost_equal(additive_blockkriging, data_check)
