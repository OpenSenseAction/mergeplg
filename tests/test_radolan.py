import numpy as np
import pandas as pd
import xarray as xr

import mergeplg as mrg


def get_test_data():
    df_stations = pd.read_csv("test_data/radolan_rain_gauge_data_new.csv", index_col=1)
    ds_radolan = xr.open_dataset("test_data/radolan_ry_data.nc")

    assert len(ds_radolan.time) == 12

    mrg.radolan.check_data_struct.check_radar_dataset_or_dataarray(
        ds_radolan, only_single_time_step=False
    )
    mrg.radolan.check_data_struct.check_station_dataframe(df_stations)

    return ds_radolan, df_stations


def test_get_test_data():
    # this is just to run the `get_test_data` function from above when running pytest
    _, _ = get_test_data()


def test_check_for_radar_coverage():
    ds_radolan, df_stations = get_test_data()
    RY_sum = ds_radolan.RY.sum(dim="time", min_count=12)

    no_radar_coverage = mrg.radolan.adjust.check_for_radar_coverage(
        x_gage=[-1500, -350, -350, 100],
        y_gage=[-4500, -4500, -4300, -4300],
        x_radar=RY_sum.x.data.flatten(),
        y_radar=RY_sum.y.data.flatten(),
        no_radar_coverage_grid=RY_sum.isnull().values,  # noqa: PD003
    )

    np.testing.assert_equal(
        no_radar_coverage,
        np.array([True, True, False, False]),
    )


def test_label_relevant_audit_interim_in_gageset_fixed_start_index():
    ds_radolan, df_stations = get_test_data()
    RY_sum = ds_radolan.RY.sum(dim="time", min_count=12)
    df_stations_with_audit_interim = (
        mrg.radolan.adjust.label_relevant_audit_interim_in_gageset(
            df_gageset_t=df_stations,
            da_radolan=RY_sum,
            start_index_in_relevant=2,
        )
    )

    assert df_stations_with_audit_interim.audit.sum() == 228
    assert df_stations_with_audit_interim.interim.sum() == 914
    audit_stations = df_stations_with_audit_interim[
        df_stations_with_audit_interim.audit
    ]
    assert audit_stations.station_id.iloc[101] == "L521"


def test_label_relevant_audit_interim_in_gageset_random_start_index():
    ds_radolan, df_stations = get_test_data()
    RY_sum = ds_radolan.RY.sum(dim="time", min_count=12)

    audit_station_id_previous = None
    N_random_runs = 5
    for i in range(N_random_runs):  # noqa: B007
        # Note that the default for start_index_in_relevant='random',
        # hence we do not set it here
        df_stations_with_audit_interim = (
            mrg.radolan.adjust.label_relevant_audit_interim_in_gageset(
                df_gageset_t=df_stations,
                da_radolan=RY_sum,
            )
        )
        audit_station_id = df_stations_with_audit_interim[
            df_stations_with_audit_interim.audit
        ].station_id.iloc[99]

        if audit_station_id_previous is None:
            audit_station_id_previous = audit_station_id
            continue
        if audit_station_id != audit_station_id_previous:
            break
        audit_station_id_previous = audit_station_id

    # This fails if all runs with random start index produced the same station_id
    assert i < N_random_runs - 1


def test_get_grid_rainfall_at_points():
    ds_radolan, df_stations = get_test_data()
    RY_sum = ds_radolan.RY.sum(dim="time", min_count=12)

    df_stations["radar_at_gauge"] = mrg.radolan.adjust.get_grid_rainfall_at_points(
        RY_sum,
        df_stations,
    )
    np.testing.assert_array_almost_equal(
        df_stations.sort_values("radar_at_gauge").radar_at_gauge.to_numpy()[-10:],
        np.array([4.69, 4.75, 4.76, 4.89, 5.73, 5.99, 6.39, 7.3, 7.57, 7.57]),
    )
    np.testing.assert_array_equal(
        df_stations.sort_values("radar_at_gauge").station_id.to_numpy()[-5:],
        np.array(["O980", "O811", "M500", "F598", "O708"], dtype=object),
    )
