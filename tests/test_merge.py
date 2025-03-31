from __future__ import annotations

import numpy as np
import poligrain as plg
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


def test_MergeDifferenceIDW():
    # CML and rain gauge overlapping sets
    da_cml_t1 = ds_cmls.isel(cml_id=[2, 1], time=[0]).R
    da_cml_t2 = ds_cmls.isel(cml_id=[1, 0], time=[0]).R
    da_gauges_t1 = ds_gauges.isel(id=[2, 1], time=[0]).R
    da_gauges_t2 = ds_gauges.isel(id=[1, 0], time=[0]).R

    # Select radar timestep
    da_rad_t = ds_rad.isel(time=[0]).R

    # Initialize highlevel-class
    merge_IDW = merge.MergeDifferenceIDW(min_observations=2)

    # Adjust field
    adjusted = merge_IDW.adjust(
        da_rad_t, da_cml=da_cml_t1, da_gauge=da_gauges_t1, method="additive"
    )

    # Check CMLs
    for cml_id in da_cml_t1.cml_id:
        merge_r = adjusted.sel(
            x=da_cml_t1.sel(cml_id=cml_id).x,
            y=da_cml_t1.sel(cml_id=cml_id).y,
        ).data
        cml_r = da_cml_t1.sel(cml_id=cml_id).data
        assert cml_r == merge_r

    # Check gauges
    for id in da_gauges_t1.id:
        merge_r = adjusted.sel(
            x=da_gauges_t1.sel(id=id).x,
            y=da_gauges_t1.sel(id=id).y,
        ).data
        gauge_r = da_gauges_t1.sel(id=id).data
        assert gauge_r == merge_r

    # Update the weights using some new links
    merge_IDW.update(
        da_rad_t,
        da_cml=da_cml_t2,
        da_gauge=da_gauges_t2,
    )

    # Test that CML names is correctly updated and sorted in the class
    assert (merge_IDW.intersect_weights.cml_id == da_cml_t2.cml_id).all()

    # Test that midpoint coordinates corresponds to x0_cml and x0_gauge
    assert (merge_IDW.x0_cml.isel(yx=1) == da_cml_t2.x).all()
    assert (merge_IDW.x0_cml.isel(yx=0) == da_cml_t2.y).all()
    assert (merge_IDW.x0_gauge.isel(yx=1) == da_gauges_t2.x).all()
    assert (merge_IDW.x0_gauge.isel(yx=0) == da_gauges_t2.y).all()

    # Define a processing function for IDW, for testing
    def keep_function(args):
        return np.where(~np.isnan(args[0]))[0]

    # Adjust field using updated geometry
    adjusted = merge_IDW.adjust(
        da_rad_t,
        da_cml=da_cml_t2,
        da_gauge=da_gauges_t2,
        method="multiplicative",
        keep_function=keep_function,
    )

    # Check that field is fit to CMLs using updated weights and
    # multiplicative adjustment
    for cml_id in da_cml_t2.cml_id:
        merge_r = adjusted.sel(
            x=da_cml_t2.sel(cml_id=cml_id).x,
            y=da_cml_t2.sel(cml_id=cml_id).y,
        ).data
        cml_r = da_cml_t2.sel(cml_id=cml_id).data
        assert cml_r == merge_r

    # Check that field is fit to gauges
    for id in da_gauges_t2.id:
        merge_r = adjusted.sel(
            x=da_gauges_t2.sel(id=id).x,
            y=da_gauges_t2.sel(id=id).y,
        ).data
        gauge_r = da_gauges_t2.sel(id=id).data
        assert gauge_r == merge_r


def test_MergeDifferenceIDW_witout_time_dim_input_data():
    merge_IDW = merge.MergeDifferenceIDW(min_observations=2)
    # test with gauge and CML data as input
    adjusted_with_time_dim = merge_IDW.adjust(
        da_rad=ds_rad.R.isel(time=[0]),
        da_cml=ds_cmls.R.isel(time=[0]),
        da_gauge=ds_gauges.R.isel(time=[0]),
        method="additive",
    ).isel(time=0)
    adjusted_without_time_dim = merge_IDW.adjust(
        da_rad=ds_rad.R.isel(time=0),
        da_cml=ds_cmls.R.isel(time=0),
        da_gauge=ds_gauges.R.isel(time=0),
        method="additive",
    )
    np.testing.assert_almost_equal(
        adjusted_with_time_dim.data, adjusted_without_time_dim.data
    )
    # test with only CML data as input (we have to test that because
    # ,at the time of writing, the relevant code is different)
    adjusted_with_time_dim = merge_IDW.adjust(
        da_rad=ds_rad.R.isel(time=[0]),
        da_cml=ds_cmls.R.isel(time=[0]),
        method="additive",
    ).isel(time=0)
    adjusted_without_time_dim = merge_IDW.adjust(
        da_rad=ds_rad.R.isel(time=0),
        da_cml=ds_cmls.R.isel(time=0),
        method="additive",
    )
    np.testing.assert_almost_equal(
        adjusted_with_time_dim.data, adjusted_without_time_dim.data
    )
    # test with only gauge data as input (at the time of writing, this
    # did not fail and was already covered by the behavior by `get_grid_at_points`
    # which uses `poligrain.spatial.GridAtPoints`)
    adjusted_with_time_dim = merge_IDW.adjust(
        da_rad=ds_rad.R.isel(time=[0]),
        da_gauge=ds_gauges.R.isel(time=[0]),
        method="additive",
    ).isel(time=0)
    adjusted_without_time_dim = merge_IDW.adjust(
        da_rad=ds_rad.R.isel(time=0),
        da_gauge=ds_gauges.R.isel(time=0),
        method="additive",
    )
    np.testing.assert_almost_equal(
        adjusted_with_time_dim.data, adjusted_without_time_dim.data
    )


def test_MergeDifferenceOrdinaryKriging():
    # CML and rain gauge overlapping sets
    da_cml_t1 = ds_cmls.isel(cml_id=[2, 1], time=[0]).R
    da_cml_t2 = ds_cmls.isel(cml_id=[1, 0], time=[0]).R
    da_gauges_t1 = ds_gauges.isel(id=[2, 1], time=[0]).R
    da_gauges_t2 = ds_gauges.isel(id=[1, 0], time=[0]).R

    # Select radar timestep
    da_rad_t = ds_rad.isel(time=[0]).R

    # Initialize highlevel-class
    merge_BK = merge.MergeDifferenceOrdinaryKriging(
        discretization=8,
        min_observations=2,
    )

    # Update geometry to set1
    merge_BK.update(
        da_rad_t,
        da_cml=da_cml_t1,
        da_gauge=da_gauges_t1,
    )

    # Test that CML names is correctly updated and sorted in the class
    assert (merge_BK.intersect_weights.cml_id == da_cml_t1.cml_id).all()

    # Test that cml midpoint and end coordinates corresponds to x0_cml
    assert (merge_BK.x0_cml.isel(yx=1, discretization=0) == da_cml_t1.site_0_x).all()
    assert (merge_BK.x0_cml.isel(yx=1, discretization=4) == da_cml_t1.x).all()
    assert (merge_BK.x0_cml.isel(yx=1, discretization=8) == da_cml_t1.site_1_x).all()
    assert (merge_BK.x0_cml.isel(yx=0, discretization=0) == da_cml_t1.site_0_y).all()
    assert (merge_BK.x0_cml.isel(yx=0, discretization=4) == da_cml_t1.y).all()
    assert (merge_BK.x0_cml.isel(yx=0, discretization=8) == da_cml_t1.site_1_y).all()

    # Test that gauge midpoint coordinates corresponds to x0_gauge
    assert (merge_BK.x0_gauge.isel(yx=1) == da_gauges_t1.x).all()
    assert (merge_BK.x0_gauge.isel(yx=0) == da_gauges_t1.y).all()

    # Test field adjustment, assuming default parameters
    adjusted = merge_BK.adjust(
        da_rad_t,
        da_cml=da_cml_t1,
        da_gauge=da_gauges_t1,
        variogram_parameters={"sill": 1, "range": 5000, "nugget": 0},
    )

    # test that the adjusted field is the same as first run
    data_check = np.array(
        [
            [
                [4.340136, 4.5248254, 5.4123771, 6.4460347],
                [4.4983906, 4.7919054, 5.9131012, 7.3120895],
                [5.0027923, 5.0, 7.1569707, 8.0085984],
                [5.968384, 6.9025046, 7.9821636, 9.0],
            ]
        ]
    )

    np.testing.assert_almost_equal(adjusted, data_check)

    # calculate the adjusted field along CMLs
    intersect_weights = plg.spatial.calc_sparse_intersect_weights_for_several_cmls(
        x1_line=da_cml_t1.site_0_x.data,
        y1_line=da_cml_t1.site_0_y.data,
        x2_line=da_cml_t1.site_1_x.data,
        y2_line=da_cml_t1.site_1_y.data,
        cml_id=da_cml_t1.cml_id.data,
        x_grid=adjusted.x_grid.data,
        y_grid=adjusted.y_grid.data,
        grid_point_location="center",
    )
    adjusted_at_cmls = plg.spatial.get_grid_time_series_at_intersections(
        grid_data=adjusted,
        intersect_weights=intersect_weights,
    )

    # As the block kriging adjustment uses the variogram to fit the rainfall field,
    # the get_grid_time_series_at_intersections uses the grid intersections and
    # the grid is discretized, the adjusted radar along will not perfectly fit the
    # CML observation. Thus we only test up to a certain decimal place.

    np.testing.assert_almost_equal(
        adjusted_at_cmls.data.ravel(),
        da_cml_t1.data.ravel(),
        decimal=-1,  # not very precise, but decent
    )

    # Test adjusted field at rain gauges
    for gauge_id in da_gauges_t1.id:
        merge_r = adjusted.sel(
            x=da_gauges_t1.sel(id=gauge_id).x.data,
            y=da_gauges_t1.sel(id=gauge_id).y.data,
        ).data
        gauge_r = da_gauges_t1.sel(id=gauge_id).data
        np.testing.assert_almost_equal(
            merge_r,
            gauge_r,
        )

    # Update the weights using some new links
    merge_BK.update(
        da_rad_t,
        da_cml=da_cml_t2,
        da_gauge=da_gauges_t2,
    )

    # Test that CML names is correctly updated and sorted in the class
    assert (merge_BK.intersect_weights.cml_id == da_cml_t2.cml_id).all()

    # Test that cml midpoint and end coordinates corresponds to x0_cml
    assert (merge_BK.x0_cml.isel(yx=1, discretization=0) == da_cml_t2.site_0_x).all()
    assert (merge_BK.x0_cml.isel(yx=1, discretization=4) == da_cml_t2.x).all()
    assert (merge_BK.x0_cml.isel(yx=1, discretization=8) == da_cml_t2.site_1_x).all()
    assert (merge_BK.x0_cml.isel(yx=0, discretization=0) == da_cml_t2.site_0_y).all()
    assert (merge_BK.x0_cml.isel(yx=0, discretization=4) == da_cml_t2.y).all()
    assert (merge_BK.x0_cml.isel(yx=0, discretization=8) == da_cml_t2.site_1_y).all()

    # Test that gauge midpoint coordinates corresponds to x0_gauge
    assert (merge_BK.x0_gauge.isel(yx=1) == da_gauges_t2.x).all()
    assert (merge_BK.x0_gauge.isel(yx=0) == da_gauges_t2.y).all()

    # Adjust field using multiplicative adjustment
    def keep_function(args):
        return np.where(~np.isnan(args[0]))[0]

    adjusted = merge_BK.adjust(
        da_rad_t,
        da_cml=da_cml_t2,
        da_gauge=da_gauges_t2,
        method="multiplicative",
        variogram_parameters={"sill": 1, "range": 5000, "nugget": 0},
        keep_function=keep_function,
    )

    # test that the adjusted field is the same as first run
    data_check = np.array(
        [
            [
                [3.2565398, 3.6323949, 3.497322, 3.1632428],
                [3.779366, 3.479784, 3.1158587, 2.8110064],
                [4.4445502, 5.0, 1.0, 2.3031289],
                [4.2632584, 3.7582347, 2.4501001, 2.2145414],
            ]
        ]
    )

    np.testing.assert_almost_equal(adjusted, data_check)

    # calculate the adjusted field along CMLs
    intersect_weights = plg.spatial.calc_sparse_intersect_weights_for_several_cmls(
        x1_line=da_cml_t2.site_0_x.data,
        y1_line=da_cml_t2.site_0_y.data,
        x2_line=da_cml_t2.site_1_x.data,
        y2_line=da_cml_t2.site_1_y.data,
        cml_id=da_cml_t2.cml_id.data,
        x_grid=adjusted.x_grid.data,
        y_grid=adjusted.y_grid.data,
        grid_point_location="center",
    )
    adjusted_at_cmls = plg.spatial.get_grid_time_series_at_intersections(
        grid_data=adjusted,
        intersect_weights=intersect_weights,
    )

    # As the block kriging adjustment uses the variogram to fit the rainfall field,
    # the get_grid_time_series_at_intersections uses the grid intersections and
    # the grid is discretized, the adjusted radar along will not perfectly fit the
    # CML observation. Thus we only test up to a certain decimal place.
    np.testing.assert_almost_equal(
        adjusted_at_cmls.data.ravel(),
        da_cml_t2.data.ravel(),
        decimal=-1,  # not very precise, but decent
    )

    # Test adjusted field at rain gauges
    for gauge_id in da_gauges_t2.id:
        merge_r = adjusted.sel(
            x=da_gauges_t2.sel(id=gauge_id).x.data,
            y=da_gauges_t2.sel(id=gauge_id).y.data,
        ).data
        gauge_r = da_gauges_t2.sel(id=gauge_id).data
        np.testing.assert_almost_equal(
            merge_r,
            gauge_r,
        )


def test_MergeDifferenceOrdinaryKriging_witout_time_dim_input_data():
    merge_OK = merge.MergeDifferenceOrdinaryKriging(min_observations=2)
    # test with gauge and CML data as input
    adjusted_with_time_dim = merge_OK.adjust(
        da_rad=ds_rad.R.isel(time=[0]),
        da_cml=ds_cmls.R.isel(time=[0]),
        da_gauge=ds_gauges.R.isel(time=[0]),
        method="additive",
    ).isel(time=0)
    adjusted_without_time_dim = merge_OK.adjust(
        da_rad=ds_rad.R.isel(time=0),
        da_cml=ds_cmls.R.isel(time=0),
        da_gauge=ds_gauges.R.isel(time=0),
        method="additive",
    )
    np.testing.assert_almost_equal(
        adjusted_with_time_dim.data, adjusted_without_time_dim.data
    )
    # test with only CML data as input (we have to test that because
    # ,at the time of writing, the relevant code is different)
    adjusted_with_time_dim = merge_OK.adjust(
        da_rad=ds_rad.R.isel(time=[0]),
        da_cml=ds_cmls.R.isel(time=[0]),
        method="additive",
    ).isel(time=0)
    adjusted_without_time_dim = merge_OK.adjust(
        da_rad=ds_rad.R.isel(time=0),
        da_cml=ds_cmls.R.isel(time=0),
        method="additive",
    )
    np.testing.assert_almost_equal(
        adjusted_with_time_dim.data, adjusted_without_time_dim.data
    )
    # test with only gauge data as input (at the time of writing, this
    # did not fail and was already covered by the behavior by `get_grid_at_points`
    # which uses `poligrain.spatial.GridAtPoints`)
    adjusted_with_time_dim = merge_OK.adjust(
        da_rad=ds_rad.R.isel(time=[0]),
        da_gauge=ds_gauges.R.isel(time=[0]),
        method="additive",
    ).isel(time=0)
    adjusted_without_time_dim = merge_OK.adjust(
        da_rad=ds_rad.R.isel(time=0),
        da_gauge=ds_gauges.R.isel(time=0),
        method="additive",
    )
    np.testing.assert_almost_equal(
        adjusted_with_time_dim.data, adjusted_without_time_dim.data
    )


def test_MergeBlockKrigingExternalDrift():
    # CML and rain gauge overlapping sets
    da_cml_t1 = ds_cmls.isel(cml_id=[2, 1], time=[0]).R
    da_cml_t2 = ds_cmls.isel(cml_id=[1, 0], time=[0]).R
    da_gauges_t1 = ds_gauges.isel(id=[2, 1], time=[0]).R
    da_gauges_t2 = ds_gauges.isel(id=[1, 0], time=[0]).R

    # Select radar timestep
    da_rad_t = ds_rad.isel(time=[0]).R

    # Initialize highlevel-class
    merge_KED = merge.MergeKrigingExternalDrift(
        discretization=8,
        min_observations=1,
    )

    # Update geometry to set1
    merge_KED.update(
        da_rad_t,
        da_cml=da_cml_t1,
        da_gauge=da_gauges_t1,
    )

    # Test that CML names is correctly updated and sorted in the class
    assert (merge_KED.intersect_weights.cml_id == da_cml_t1.cml_id).all()

    # Test that cml midpoint and end coordinates corresponds to x0_cml
    assert (merge_KED.x0_cml.isel(yx=1, discretization=0) == da_cml_t1.site_0_x).all()
    assert (merge_KED.x0_cml.isel(yx=1, discretization=4) == da_cml_t1.x).all()
    assert (merge_KED.x0_cml.isel(yx=1, discretization=8) == da_cml_t1.site_1_x).all()
    assert (merge_KED.x0_cml.isel(yx=0, discretization=0) == da_cml_t1.site_0_y).all()
    assert (merge_KED.x0_cml.isel(yx=0, discretization=4) == da_cml_t1.y).all()
    assert (merge_KED.x0_cml.isel(yx=0, discretization=8) == da_cml_t1.site_1_y).all()

    # Test that gauge midpoint coordinates corresponds to x0_gauge
    assert (merge_KED.x0_gauge.isel(yx=1) == da_gauges_t1.x).all()
    assert (merge_KED.x0_gauge.isel(yx=0) == da_gauges_t1.y).all()

    # Test that adjusted field is the same
    adjusted = merge_KED.adjust(
        da_rad_t,
        da_cml=da_cml_t1,
        da_gauge=da_gauges_t1,
        variogram_parameters={"sill": 1, "range": 5000, "nugget": 0},
    )

    # test that the adjusted field is the same as first run
    data_check = np.array(
        [
            [
                [4.340136, 4.5248254, 5.4123771, 6.4460347],
                [4.4983906, 4.7919054, 5.9131012, 7.3120895],
                [5.0027923, 5.0, 7.1569707, 8.0085984],
                [5.968384, 6.9025046, 7.9821636, 9.0],
            ]
        ]
    )

    np.testing.assert_almost_equal(adjusted, data_check)

    # calculate the adjusted field along CMLs
    intersect_weights = plg.spatial.calc_sparse_intersect_weights_for_several_cmls(
        x1_line=da_cml_t1.site_0_x.data,
        y1_line=da_cml_t1.site_0_y.data,
        x2_line=da_cml_t1.site_1_x.data,
        y2_line=da_cml_t1.site_1_y.data,
        cml_id=da_cml_t1.cml_id.data,
        x_grid=adjusted.x_grid.data,
        y_grid=adjusted.y_grid.data,
        grid_point_location="center",
    )
    adjusted_at_cmls = plg.spatial.get_grid_time_series_at_intersections(
        grid_data=adjusted,
        intersect_weights=intersect_weights,
    )

    # As the block kriging adjustment uses the variogram to fit the rainfall field,
    # the get_grid_time_series_at_intersections uses the grid intersections and
    # the grid is discretized, the adjusted radar along will not perfectly fit the
    # CML observation. Thus we only test up to a certain decimal place.
    np.testing.assert_almost_equal(
        adjusted_at_cmls.data.ravel(),
        da_cml_t1.data.ravel(),
        decimal=-1,  # not very precise, but decent
    )

    # Test adjusted field at rain gauges
    for gauge_id in da_gauges_t1.id:
        merge_r = adjusted.sel(
            x=da_gauges_t1.sel(id=gauge_id).x.data,
            y=da_gauges_t1.sel(id=gauge_id).y.data,
        ).data
        gauge_r = da_gauges_t1.sel(id=gauge_id).data
        np.testing.assert_almost_equal(
            merge_r,
            gauge_r,
        )

    # Update the weights using some new links
    merge_KED.update(
        da_rad_t,
        da_cml=da_cml_t2,
        da_gauge=da_gauges_t2,
    )

    # Test that CML names is correctly updated and sorted in the class
    assert (merge_KED.intersect_weights.cml_id == da_cml_t2.cml_id).all()

    # Test that cml midpoint and end coordinates corresponds to x0_cml
    assert (merge_KED.x0_cml.isel(yx=1, discretization=0) == da_cml_t2.site_0_x).all()
    assert (merge_KED.x0_cml.isel(yx=1, discretization=4) == da_cml_t2.x).all()
    assert (merge_KED.x0_cml.isel(yx=1, discretization=8) == da_cml_t2.site_1_x).all()
    assert (merge_KED.x0_cml.isel(yx=0, discretization=0) == da_cml_t2.site_0_y).all()
    assert (merge_KED.x0_cml.isel(yx=0, discretization=4) == da_cml_t2.y).all()
    assert (merge_KED.x0_cml.isel(yx=0, discretization=8) == da_cml_t2.site_1_y).all()

    # Test that gauge midpoint coordinates corresponds to x0_gauge
    assert (merge_KED.x0_gauge.isel(yx=1) == da_gauges_t2.x).all()
    assert (merge_KED.x0_gauge.isel(yx=0) == da_gauges_t2.y).all()

    # Adjust field
    adjusted = merge_KED.adjust(
        da_rad_t,
        da_cml=da_cml_t2,
        da_gauge=da_gauges_t2,
        variogram_parameters={"sill": 1, "range": 5000, "nugget": 0},
    )

    # test that the adjusted field is the same as first run
    data_check = np.array(
        [
            [
                [3.2565398, 3.6323949, 3.497322, 3.1632428],
                [3.779366, 3.479784, 3.1158587, 2.8110064],
                [4.4445502, 5.0, 1.0, 2.3031289],
                [4.2632584, 3.7582347, 2.4501001, 2.2145414],
            ]
        ]
    )

    np.testing.assert_almost_equal(adjusted, data_check)

    # calculate the adjusted field along CMLs
    intersect_weights = plg.spatial.calc_sparse_intersect_weights_for_several_cmls(
        x1_line=da_cml_t2.site_0_x.data,
        y1_line=da_cml_t2.site_0_y.data,
        x2_line=da_cml_t2.site_1_x.data,
        y2_line=da_cml_t2.site_1_y.data,
        cml_id=da_cml_t2.cml_id.data,
        x_grid=adjusted.x_grid.data,
        y_grid=adjusted.y_grid.data,
        grid_point_location="center",
    )
    adjusted_at_cmls = plg.spatial.get_grid_time_series_at_intersections(
        grid_data=adjusted,
        intersect_weights=intersect_weights,
    )

    # As the block kriging adjustment uses the variogram to fit the rainfall field,
    # the get_grid_time_series_at_intersections uses the grid intersections and
    # the grid is discretized, the adjusted radar along will not perfectly fit the
    # CML observation. Thus we only test up to a certain decimal place.
    np.testing.assert_almost_equal(
        adjusted_at_cmls.data.ravel(),
        da_cml_t2.data.ravel(),
        decimal=-1,  # not very precise, but decent
    )

    # Test adjusted field at rain gauges
    for gauge_id in da_gauges_t2.id:
        merge_r = adjusted.sel(
            x=da_gauges_t2.sel(id=gauge_id).x.data,
            y=da_gauges_t2.sel(id=gauge_id).y.data,
        ).data
        gauge_r = da_gauges_t2.sel(id=gauge_id).data
        np.testing.assert_almost_equal(
            merge_r,
            gauge_r,
        )

    # Test that providing only rain gauge adjusts only at rain gauge
    adjusted = merge_KED.adjust(
        da_rad_t,
        da_cml=None,
        da_gauge=da_gauges_t2,
        variogram_parameters={"sill": 1, "range": 5000, "nugget": 0},
    )

    # Test adjusted field at rain gauges
    for gauge_id in da_gauges_t2.id:
        merge_r = adjusted.sel(
            x=da_gauges_t2.sel(id=gauge_id).x.data,
            y=da_gauges_t2.sel(id=gauge_id).y.data,
        ).data
        gauge_r = da_gauges_t2.sel(id=gauge_id).data
        np.testing.assert_almost_equal(
            merge_r,
            gauge_r,
        )

    # Test that providing only cml adjusts at cml
    adjusted = merge_KED.adjust(
        da_rad_t,
        da_cml=da_cml_t2,
        da_gauge=None,
        variogram_parameters={"sill": 1, "range": 5000, "nugget": 0},
    )

    # calculate the adjusted field along CMLs
    intersect_weights = plg.spatial.calc_sparse_intersect_weights_for_several_cmls(
        x1_line=da_cml_t2.site_0_x.data,
        y1_line=da_cml_t2.site_0_y.data,
        x2_line=da_cml_t2.site_1_x.data,
        y2_line=da_cml_t2.site_1_y.data,
        cml_id=da_cml_t2.cml_id.data,
        x_grid=adjusted.x_grid.data,
        y_grid=adjusted.y_grid.data,
        grid_point_location="center",
    )
    adjusted_at_cmls = plg.spatial.get_grid_time_series_at_intersections(
        grid_data=adjusted,
        intersect_weights=intersect_weights,
    )

    # Test cml almost equal
    np.testing.assert_almost_equal(
        adjusted_at_cmls.data.ravel(),
        da_cml_t2.data.ravel(),
        decimal=0,  # not very precise, but decent
    )


def test_MergeDifferenceKED_witout_time_dim_input_data():
    merge_OK = merge.MergeKrigingExternalDrift(min_observations=2)
    # test with gauge and CML data as input
    adjusted_with_time_dim = merge_OK.adjust(
        da_rad=ds_rad.R.isel(time=[0]),
        da_cml=ds_cmls.R.isel(time=[0]),
        da_gauge=ds_gauges.R.isel(time=[0]),
    ).isel(time=0)
    adjusted_without_time_dim = merge_OK.adjust(
        da_rad=ds_rad.R.isel(time=0),
        da_cml=ds_cmls.R.isel(time=0),
        da_gauge=ds_gauges.R.isel(time=0),
    )
    np.testing.assert_almost_equal(
        adjusted_with_time_dim.data, adjusted_without_time_dim.data
    )
    # test with only CML data as input (we have to test that because
    # ,at the time of writing, the relevant code is different)
    adjusted_with_time_dim = merge_OK.adjust(
        da_rad=ds_rad.R.isel(time=[0]),
        da_cml=ds_cmls.R.isel(time=[0]),
    ).isel(time=0)
    adjusted_without_time_dim = merge_OK.adjust(
        da_rad=ds_rad.R.isel(time=0),
        da_cml=ds_cmls.R.isel(time=0),
    )
    np.testing.assert_almost_equal(
        adjusted_with_time_dim.data, adjusted_without_time_dim.data
    )
    # test with only gauge data as input (at the time of writing, this
    # did not fail and was already covered by the behavior by `get_grid_at_points`
    # which uses `poligrain.spatial.GridAtPoints`)
    adjusted_with_time_dim = merge_OK.adjust(
        da_rad=ds_rad.R.isel(time=[0]),
        da_gauge=ds_gauges.R.isel(time=[0]),
    ).isel(time=0)
    adjusted_without_time_dim = merge_OK.adjust(
        da_rad=ds_rad.R.isel(time=0),
        da_gauge=ds_gauges.R.isel(time=0),
    )
    np.testing.assert_almost_equal(
        adjusted_with_time_dim.data, adjusted_without_time_dim.data
    )
