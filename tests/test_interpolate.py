from __future__ import annotations

import numpy as np
import pykrige
import xarray as xr

from mergeplg import interpolate

ds_cmls = xr.Dataset(
    data_vars={
        "R": (("cml_id", "time"), np.reshape(np.arange(1, 13), (3, 4))),
    },
    coords={
        "cml_id": ("cml_id", ["cml1", "cml2", "cml3"]),
        "time": ("time", np.arange(0, 4)),
        "site_0_x": ("cml_id", [-1, 0, 0]),
        "site_0_y": ("cml_id", [-1, -1, 2]),
        "site_1_x": ("cml_id", [1, 2, 2]),
        "site_1_y": ("cml_id", [1, 1, 0]),
        "site_0_lon": ("cml_id", [-1, 0, 0]),
        "site_0_lat": ("cml_id", [-1, -1, 2]),
        "site_1_lon": ("cml_id", [1, 2, 2]),
        "site_1_lat": ("cml_id", [1, 1, 0]),
        "x": ("cml_id", [0, 1, 1]),
        "y": ("cml_id", [0, 0, 1]),
        "length": ("cml_id", [2 * np.sqrt(2), 2 * np.sqrt(2), 2 * np.sqrt(2)]),
    },
)

ds_gauges = xr.Dataset(
    data_vars={
        "R": (("id", "time"), np.reshape(np.arange(1, 13), (3, 4))),
    },
    coords={
        "id": ("id", ["gauge1", "gauge2", "gauge3"]),
        "time": ("time", np.arange(0, 4)),
        "lon": ("id", [1, 0, 2]),
        "lat": ("id", [1, 1, 2]),
        "x": ("id", [1, 0, 2]),
        "y": ("id", [1, 1, 2]),
    },
)

ds_rad = xr.Dataset(
    data_vars={"R": (("time", "y", "x"), np.ones([4, 4, 4]))},
    coords={
        "x": ("x", [-1, 0, 1, 2]),
        "y": ("y", [-1, 0, 1, 2]),
        "time": ("time", [0, 1, 2, 3]),
        "lon": (("y", "x"), np.meshgrid([-1, 0, 1, 2], [-1, 0, 1, 2])[0]),
        "lat": (("y", "x"), np.meshgrid([-1, 0, 1, 2], [-1, 0, 1, 2])[1]),
        "x_grid": (("y", "x"), np.meshgrid([-1, 0, 1, 2], [-1, 0, 1, 2])[0]),
        "y_grid": (("y", "x"), np.meshgrid([-1, 0, 1, 2], [-1, 0, 1, 2])[1]),
    },
)


def test_blockkriging_vs_pykrige():
    # CML and rain gauge overlapping sets
    da_cml_t1 = ds_cmls.isel(cml_id=[2, 1], time=[0]).R
    da_gauges_t1 = ds_gauges.isel(id=[2, 1], time=[0]).R

    da_grid = ds_rad.isel(time=[0]).R

    # Initialize highlevel-class
    interpolate_krig = interpolate.InterpolateOrdinaryKriging(min_observations=2)

    variogram_model = "exponential"
    variogram_parameters = {"sill": 1, "range": 2, "nugget": 0.5}

    # Interpolate field
    interp_field = interpolate_krig.interpolate(
        da_grid,
        da_cml=da_cml_t1,
        da_gauge=da_gauges_t1,
        variogram_model=variogram_model,
        variogram_parameters=variogram_parameters,
        nnear=False,
        full_line=False,
    )

    # Get ground observations and x0 geometry
    obs, x0 = interpolate_krig.get_obs_x0_(da_cml=da_cml_t1, da_gauge=da_gauges_t1)

    # Setup pykrige using midpoint of CMLs as reference
    ok = pykrige.OrdinaryKriging(
        x0[:, 1, int(x0.shape[2] / 2)],  # x-midpoint coordinate
        x0[:, 0, int(x0.shape[2] / 2)],  # y-midpoint coordinate
        obs.ravel(),
        variogram_model=variogram_model,
        variogram_parameters=variogram_parameters,
        pseudo_inv=True,
        exact_values=False,  # Account for nugget when predicting
    )

    z, ss = ok.execute(
        "points",
        da_grid.y_grid.data.ravel().astype(float),
        da_grid.x_grid.data.ravel().astype(float),
    )

    interp_field_pykrige = [z.reshape(da_grid.x_grid.shape)]

    np.testing.assert_almost_equal(interp_field_pykrige, interp_field)


def test_idw_interpolate_with_data_without_time_dim():
    idw_interpolator = interpolate.InterpolateIDW(min_observations=2)
    # Make sure that the interpolation works with and without time dimension
    # in the supplied data arrays and that resulta are the same
    R_grid_idw_no_time_dim = idw_interpolator.interpolate(
        da_grid=ds_rad.R.isel(time=0),
        da_cml=ds_cmls.R.isel(time=0),
        p=3,
        idw_method="standard",
    )
    R_grid_idw_with_time_dim = idw_interpolator.interpolate(
        da_grid=ds_rad.R.isel(time=[0]),
        da_cml=ds_cmls.R.isel(time=[0]),
        p=3,
        idw_method="standard",
    ).isel(time=0)
    np.testing.assert_almost_equal(
        R_grid_idw_no_time_dim.data, R_grid_idw_with_time_dim.data
    )
    # Same with gauge data ds_geauges instead of ds_cmls
    R_grid_idw_no_time_dim = idw_interpolator.interpolate(
        da_grid=ds_rad.R.isel(time=0),
        da_gauge=ds_gauges.R.isel(time=0),
        p=3,
        idw_method="standard",
    )
    R_grid_idw_with_time_dim = idw_interpolator.interpolate(
        da_grid=ds_rad.R.isel(time=[0]),
        da_gauge=ds_gauges.R.isel(time=[0]),
        p=3,
        idw_method="standard",
    ).isel(time=0)
    np.testing.assert_almost_equal(
        R_grid_idw_no_time_dim.data, R_grid_idw_with_time_dim.data
    )


def test_ok_interpolate_with_data_without_time_dim():
    ok_interpolator = interpolate.InterpolateOrdinaryKriging(min_observations=2)
    # Make sure that the interpolation works with and without time dimension
    # in the supplied data arrays and that resulta are the same
    R_grid_idw_no_time_dim = ok_interpolator.interpolate(
        da_grid=ds_rad.R.isel(time=0),
        da_cml=ds_cmls.R.isel(time=0),
    )
    R_grid_idw_with_time_dim = ok_interpolator.interpolate(
        da_grid=ds_rad.R.isel(time=[0]),
        da_cml=ds_cmls.R.isel(time=[0]),
    ).isel(time=0)
    np.testing.assert_almost_equal(
        R_grid_idw_no_time_dim.data, R_grid_idw_with_time_dim.data
    )
    # Same with gauge data ds_geauges instead of ds_cmls
    R_grid_idw_no_time_dim = ok_interpolator.interpolate(
        da_grid=ds_rad.R.isel(time=0),
        da_gauge=ds_gauges.R.isel(time=0),
    )
    R_grid_idw_with_time_dim = ok_interpolator.interpolate(
        da_grid=ds_rad.R.isel(time=[0]),
        da_gauge=ds_gauges.R.isel(time=[0]),
    ).isel(time=0)
    np.testing.assert_almost_equal(
        R_grid_idw_no_time_dim.data, R_grid_idw_with_time_dim.data
    )
