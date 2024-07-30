import numpy as np
import wradlib
import xarray as xr
from scipy import ndimage

from . import check_data_struct, idw


def label_relevant_audit_interim_in_gageset(
    df_gageset_t, da_radolan, start_index_in_relevant="random"
):
    """Add label for relevant, audit and interim stations to Dataframe of gageset

    Parameters
    ----------

    """

    df_gageset_t = df_gageset_t.copy()

    # Find relevant stations
    no_radar_coverage = check_for_radar_coverage(
        x_gage=df_gageset_t.x.values,
        y_gage=df_gageset_t.y.values,
        x_radar=da_radolan.x.values,
        y_radar=da_radolan.y.values,
        no_radar_coverage_grid=da_radolan.isnull().values,
    )

    df_gageset_t.loc[:, "no_radar_coverage"] = no_radar_coverage

    df_gageset_t.loc[:, "relevant"] = (
        ~df_gageset_t.no_radar_coverage & ~df_gageset_t.rainfall_amount.isna()
    )

    if start_index_in_relevant == "random":
        start_index = np.random.randint(df_gageset_t.relevant.sum())
    elif isinstance(start_index_in_relevant, int):
        start_index = start_index_in_relevant
    else:
        raise TypeError("`start_index_in_relevant` has to be an integer or 'random'")

    # Find audit stations
    index_control_stations = get_audit_station_indicies(
        x_gage=df_gageset_t.x.values,
        y_gage=df_gageset_t.y.values,
        index_relevant_stations=np.where(df_gageset_t.relevant)[0],
        start_index=start_index,
    )

    audit = np.zeros(len(df_gageset_t), dtype="bool")
    audit[index_control_stations] = True
    df_gageset_t.loc[:, "audit"] = audit
    df_gageset_t["interim"] = ~df_gageset_t.audit & df_gageset_t.relevant

    return df_gageset_t


def check_for_radar_coverage(x_gage, y_gage, x_radar, y_radar, no_radar_coverage_grid):
    """Check if gauge coordinates are in region with radar coverage

    Note that this only works for a equidistant radar grid which is
    defined by one x and one y vector, because it uses the min and
    max value to define a bounding box of the radar grid.

    Parameters
    ----------

    x_gage: array of x coordiantes for n gauges
    y_gage: ...
    x_radar: array of x coordinates of radar grid
    y_radar: ...
    no_radar_coverage_grid: 2d bool matrix with True for entries with no radar coverage

    Returns
    -------

    no_radar_coverage:
        bool array with same length as x_gage. True for coordinates
        with no radar coverage.

    """

    x_grid, y_grid = np.meshgrid(x_radar, y_radar)
    xy_radar = np.array(list(zip(x_grid.flatten(), y_grid.flatten())))

    raw_at_obs = wradlib.adjust.RawAtObs(
        obs_coords=np.array(list(zip(x_gage, y_gage))),
        raw_coords=xy_radar,
        nnear=1,
    )
    # True for points all points where the closest RADOLAN grid point has coverage.
    # Note that this is also true for points outside of the RADOLAN grid, which have
    # no radar coverage
    closest_point_not_covered = raw_at_obs(raw=no_radar_coverage_grid.flatten())

    # Add requirement to be insided the coordinate range of the RADOLAN grid
    no_radar_coverage = (
        closest_point_not_covered
        | (x_gage < x_radar.min())
        | (x_gage > x_radar.max())
        | (y_gage < y_radar.min())
        | (y_gage > y_radar.max())
    )
    return no_radar_coverage


def get_audit_station_indicies(
    x_gage, y_gage, index_relevant_stations, start_index, print_info=False
):
    """Get indices to select audit stations based on a list of cooridates of stations

    Start at one relevant station. Find 5 nearest neigbors. Save indices
    all of these neibors in a list of visited stations. Selecte station
    that is farest away. Start over until less than 6 stations are available.


    Parameters
    ----------
    x_gage: array of x coordinates of all station (e.g. all from gageset.tab)
    y_gage: ...
    index_relevant_stations:
        array with indices of relevant stations in x_gage and y_gage. If you
        have a dataframe with a "relevant" column, you can produce this array
        using `np.where(df_gageset.relevant)[0]`.
    start_index: int
        Index which indicates which entry in the array of `index_relevant_stations`
        to use as start index for the audit station search.

    Returns
    -------
    index_control_stations:
        List if indices of the audit stations. Indices are based on the array
        of x_gage and y_gage.

    """

    from scipy.spatial import cKDTree as KDTree

    index_control_stations = []
    index_already_visited_stations = []

    start_station_index = index_relevant_stations[start_index]

    xy_all_stations = np.array(list(zip(x_gage, y_gage)))

    while True:
        xy_current_station = xy_all_stations[start_station_index]

        index_not_yet_visited_stations = set(index_relevant_stations) - set(
            index_already_visited_stations
        )
        xy_not_yet_visited_stations = xy_all_stations[
            list(index_not_yet_visited_stations), :
        ]

        if len(index_not_yet_visited_stations) <= 5:
            break

        tree = KDTree(xy_not_yet_visited_stations)
        distance_neighbors, index_neighbors = tree.query(xy_current_station, k=6)
        index_neighbors = np.array(list(index_not_yet_visited_stations))[
            index_neighbors
        ]

        # Go from 2D array to 1D array and cut off first entry
        # which is the site of the current station
        distance_neighbors = distance_neighbors[1:]
        index_neighbors = index_neighbors[1:]

        index_already_visited_stations += list(index_neighbors)

        # Take the furthers of the five neighbors
        index_selected_neighbor = index_neighbors[-1]
        index_control_stations.append(index_selected_neighbor)

        start_station_index = index_selected_neighbor

    return index_control_stations


def get_grid_rainfall_at_points(da_grid, df_stations, nnear=9, stat="best"):
    """Get a gridded rainfall value at a certain locations

    Parameters
    ----------

    da_grid : xarray.DataArray
        Must contain a 1D 'x' and 'y' variable with coordinates of the grid.
        Must also contain the variable 'rainfall_amount'.
    df_station : pandas.Dataframe
        Must contain a colum 'x' and 'y' with station coordinates.
        Must also contain a column 'rainfall_amount'.
    nnear : int
        Number of surrounding grid pixels to consider
    stat : ['mean', 'median', 'best']
        How to choose value in the nnear grid cells.

    Returns
    -------

    array of grid values at (or near) station location

    """

    x_grid, y_grid = np.meshgrid(da_grid.x, da_grid.y)
    xy_radar = np.array(list(zip(x_grid.flatten(), y_grid.flatten())))

    raw_at_obs_adjuster = wradlib.adjust.RawAtObs(
        obs_coords=df_stations.loc[:, ["x", "y"]].values,
        raw_coords=xy_radar,
        nnear=nnear,
        stat=stat,
    )

    return raw_at_obs_adjuster(
        obs=df_stations.loc[:, "rainfall_amount"].values,
        raw=da_grid.values.flatten(),
    )


def interpolate_station_values(
    df_stations,
    col_name,
    ds_grid,
    station_ids_to_exclude=[],
    nnear=20,
    p=2,
    max_distance=60,
    idw_method="radolan",
    fill_value=np.nan,
):
    #check_data_struct.check_station_dataframe(df_stations, only_single_time_step=True)

    df_stations_temp = df_stations[~df_stations.station_id.isin(station_ids_to_exclude)]

    z = df_stations_temp[col_name]

    x_grid, y_grid = np.meshgrid(ds_grid.x, ds_grid.y)
    xy_radar = np.array(list(zip(x_grid.flatten(), y_grid.flatten())))

    idw_interpolator = idw.Invdisttree(
        df_stations_temp.loc[~z.isnull(), ["x", "y"]].values
    )
    if np.all(np.isnan(z)):
        zi = np.empty_like(x_grid)
        zi[:] = np.nan
    else:
        zi = idw_interpolator(
            q=xy_radar,
            z=z[~z.isnull()],
            nnear=nnear,
            p=p,
            idw_method=idw_method,
            max_distance=max_distance,
        )

    zi[np.isnan(zi)] = fill_value

    da = xr.DataArray(coords=[ds_grid.y, ds_grid.x], data=zi.reshape(x_grid.shape))
    return da


def bogra_like_smoothing(radar_img, max_iterations=100, max_allowed_relative_diff=3):
    """Apply BOGRA to smooth data"""

    kernel_diff = np.array(
        [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
    )
    kernel_mean = np.array(
        [[1 / 8, 1 / 8, 1 / 8], [1 / 8, 0, 1 / 8], [1 / 8, 1 / 8, 1 / 8]],
    )

    if isinstance(radar_img, xr.core.dataarray.DataArray):
        radar_img_filtered = radar_img.values.copy()
    else:
        radar_img_filtered = radar_img.copy()

    for i in range(max_iterations):
        highpass_3x3 = ndimage.convolve(radar_img_filtered, kernel_diff)
        relative_gradients = highpass_3x3 / np.ma.masked_array(
            radar_img_filtered, mask=np.isnan(radar_img_filtered)
        )

        if np.nanmax(relative_gradients) > max_allowed_relative_diff:
            mean_3x3 = ndimage.convolve(radar_img_filtered, kernel_mean)
            # fill in mean for positive relative diff over threshold
            radar_img_filtered[
                relative_gradients > max_allowed_relative_diff
            ] = mean_3x3[relative_gradients > max_allowed_relative_diff]
        else:
            break

    if isinstance(radar_img, xr.core.dataarray.DataArray):
        return xr.DataArray(
            data=radar_img_filtered, dims=radar_img.dims, coords=radar_img.coords
        )
    else:
        return radar_img_filtered
