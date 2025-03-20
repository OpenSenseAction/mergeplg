"""Module for merging CML and radar data."""

from __future__ import annotations

import numpy as np
import xarray as xr
import pykrige

from .radolan import idw
from mergeplg import bk_functions
from mergeplg.base import Base


class MergeDifferenceIDW(Base):
    """Merge ground and radar difference using IDW.

    Merges the provided radar field in ds_rad with gauge or CML observations
    by interpolating the difference (additive or multiplicative) 
    between the ground and radar observations using IDW. 
    """

    def __init__(
        self,
        grid_location_radar="center",
    ):
        Base.__init__(self, grid_location_radar)

    def update(self, da_rad, da_cml=None, da_gauge=None):
        """Update weights and x0 geometry for CML and gauge

        This function uses the midpoint of the CML as CML reference.
        """
        # Update x0 and radar weights
        self.update_x0_(da_cml=da_cml, da_gauge=da_gauge)
        self.update_weights_(da_rad, da_cml=da_cml, da_gauge=da_gauge)

    def adjust(
        self,
        da_rad,
        da_cml=None,
        da_gauge=None,
        p=2,
        idw_method="radolan",
        nnear=8,
        max_distance=60000,
        method='additive'
    ):
        """Adjust radar field for one time step.

        Adjust radar field for one time step. The function assumes that the
        weights are updated using the update class method.

        Parameters
        ----------
        da_rad: xarray.DataArray
            Gridded radar data. Must contain the lon and lat coordinates as
            well as the projected coordinates xs and ys as a meshgrid.
        da_cml: xarray.DataArray
            CML observations. Must contain the lat/lon coordinates for the CML
            (site_0_lon, site_0_lat, site_1_lon, site_1_lat) as well as the
            projected midpoint coordinates (x, y).
        da_gauge: xarray.DataArray
            Gauge observations. Must contain the coordinates for the rain gauge
            positions (lat, lon) as well as the projected coordinates (x, y).
        p: float
            IDW interpolation parameter
        idw_method: str
            by default "radolan"
        n_closest: int
            Number of neighbours to use for interpolation. 
        max_distance: float
            max distance allowed interpolation distance
        method: str
            Set to 'additive' to use additive approach, or 'multiplicative' to
            use the multiplicative approach.

        Returns
        -------
        da_rad_out: xarray.DataArray
            DataArray with the same structure as the ds_rad but with the CML
            adjusted radar field.
        """
        # Update weights and x0 geometry for CML and gauge
        self.update(da_rad, da_cml=da_cml, da_gauge=da_gauge)

        # Evaluate radar at cml and gauge ground positions
        rad, obs, x0 = self.get_rad_obs_x0_(da_rad, da_cml=da_cml, da_gauge=da_gauge)

        # Calculate radar-ground difference if radar observes rainfall
        if method == 'additive':
            diff = np.where(rad>0, obs- rad, np.nan)
            keep = np.where(~np.isnan(diff))[0]

        elif method == 'multiplicative':
            mask_zero = rad > 0.0
            diff = np.full_like(obs, np.nan, dtype=np.float64)
            diff[mask_zero] = obs[mask_zero]/rad[mask_zero]
            keep = np.where((~np.isnan(diff)) & (diff < np.nanquantile(diff, 0.95)))[0]

        else:
            msg = "Method must be multiplicative or additive"
            raise ValueError(msg)
        
        # Coordinates to predict
        coord_pred = np.hstack(
            [da_rad.ys.data.reshape(-1, 1), da_rad.xs.data.reshape(-1, 1)]
        )

        # IDW interpolator invdisttree
        idw_interpolator = idw.Invdisttree(x0[keep])
        interpolated = idw_interpolator(
            q=coord_pred,
            z=diff[keep],
            nnear=obs[keep].size if obs[keep].size <= nnear else nnear,
            p=p,
            idw_method=idw_method,
            max_distance=max_distance,
        ).reshape(da_rad.xs.shape)

        # Adjust radar field
        if method == 'additive':
            adjusted = interpolated + da_rad.isel(time = 0).data
        elif method == 'multiplicative':
            adjusted = interpolated*da_rad.isel(time = 0).data

        # Remove negative values
        adjusted[adjusted < 0] = 0

        return xr.DataArray(
            data=[adjusted],
            coords=da_rad.coords,
            dims=da_rad.dims
        )

class MergeDifferenceBlockKriging(Base):
    """Merge CML and radar using block kriging

    Merges the provided radar field in ds_rad with gauge or CML observations
    by interpolating the difference (additive or multiplicative) 
    between the ground and radar observations using Block Kriging.
    """

    def __init__(
        self,
        grid_location_radar="center",
        discretization=8,
    ):
        Base.__init__(self, grid_location_radar)

        # Number of discretization points along CML
        self.discretization = discretization

        # For storing variogram parameters
        self.variogram_param = None

    def update(self, da_rad, da_cml=None, da_gauge=None):
        """Update weights and x0 geometry for CML and gauge assuming block data

        This function uses the full CML geometry and makes the gauge geometry
        appear as a rain gauge.
        """
        self.update_weights_(da_rad, da_cml=da_cml, da_gauge=da_gauge)
        self.update_x0_block_(self.discretization, da_cml=da_cml, da_gauge=da_gauge)

    def adjust(
        self, 
        da_rad, 
        da_cml=None, 
        da_gauge=None, 
        variogram_model="spherical",
        variogram_parameters={"sill": 0.9, "range": 5000, "nugget": 0.1},
        nnear=8,
        max_distance=60000,
        full_line = True,
        method='additive'
    ):
        """Interpolate observations for one time step.

        Interpolates ground observations for one time step. The function assumes that the
        x0 are updated using the update class method.

        Parameters
        ----------
        da_rad: xarray.DataArray
            Gridded radar data. Must contain the lon and lat coordinates as
            well as the projected coordinates xs and ys as a meshgrid.
        da_cml: xarray.DataArray
            CML observations. Must contain the projected midpoint 
            coordinates (x, y).
        da_gauge: xarray.DataArray
            Gauge observations. Must contain the projected 
            coordinates (x, y).
        variogram_model: str
            Must be a valid variogram type in pykrige.
        variogram_parameters: str
            Must be a valid parameters corresponding to variogram_model.
        nnear: int
            Number of closest links to use for interpolation
        max_distance: float
            Largest distance allowed for including an observation.
        full_line: bool
            Wether to use the full line for block kriging. If set to false, the 
            x0 geometry is reformated to simply reflect the midpoint of the CML. 
        method: str
            Set to 'additive' to use additive approach, or 'multiplicative' to
            use the multiplicative approach.

        Returns
        -------
        da_field_out: xarray.DataArray
            DataArray with the same structure as the ds_rad but with the
            interpolated field. 
        """
        # Update weights and x0 geometry for CML and gauge
        self.update(da_rad, da_cml=da_cml, da_gauge=da_gauge)

        # Evaluate radar at cml and gauge ground positions
        rad, obs, x0 = self.get_rad_obs_x0_(da_rad, da_cml=da_cml, da_gauge=da_gauge)

        # Calculate radar-ground difference if radar observes rainfall
        if method == 'additive':
            diff = np.where(rad>0, obs- rad, np.nan)
            keep = np.where(~np.isnan(diff))[0]

        elif method == 'multiplicative':
            mask_zero = rad > 0.0
            diff = np.full_like(obs, np.nan, dtype=np.float64)
            diff[mask_zero] = obs[mask_zero]/rad[mask_zero]
            keep = np.where((~np.isnan(diff)) & (diff < np.nanquantile(diff, 0.95)))[0]

        else:
            msg = "Method must be multiplicative or additive"
            raise ValueError(msg)

        # Setup pykrige with variogram parameters provided by user
        ok = pykrige.OrdinaryKriging(
            x0[keep, 1, int(x0.shape[1] / 2)], #x-midpoint coordinate
            x0[keep, 0, int(x0.shape[1] / 2)], #y-midpoint coordinate
            diff[keep],
            variogram_model=variogram_model,
            variogram_parameters=variogram_parameters,
        )
       
        # Construct variogram using pykrige
        def variogram(h):
            return ok.variogram_function(ok.variogram_model_parameters, h)
        
        # Force interpolator to use only midpoint
        if full_line is False:
            x0 = x0[keep, :, [int(x0.shape[1] / 2)]]

        # If nnear is set to False, use all observations in kriging
        if nnear == False:
            interpolated = bk_functions.interpolate_block_kriging(
                da_rad.xs.data,
                da_rad.ys.data,
                diff[keep], 
                x0[keep],
                variogram,
            )    

        # Else do neighbourhood kriging
        else:
            interpolated = bk_functions.interpolate_neighbourhood_block_kriging(
                da_rad.xs.data,
                da_rad.ys.data,
                diff[keep], 
                x0[keep],
                variogram,
                diff[keep].size - 1 if diff[keep].size <= nnear else nnear,
            )

        # Adjust radar field
        if method == 'additive':
            adjusted = interpolated + da_rad.isel(time = 0).data
        elif method == 'multiplicative':
            adjusted = interpolated*da_rad.isel(time = 0).data


        # Remove negative values
        adjusted[adjusted < 0] = 0

        return xr.DataArray(
            data=[adjusted],
            coords=da_rad.coords,
            dims=da_rad.dims
        )

class MergeBlockKrigingExternalDrift(Base):
    """Merge CML and radar using block-kriging with external drift.

    Merges the provided radar field in ds_rad to CML and rain gauge
    observations by using a block kriging variant of kriging with external
    drift.
    """

    def __init__(
        self,
        grid_location_radar="center",
        discretization=8,
    ):
        Base.__init__(self, grid_location_radar)

        # Number of discretization points along CML
        self.discretization = discretization

    def update(self, da_rad, da_cml=None, da_gauge=None):
        """Update weights and x0 geometry for CML and gauge assuming block data

        This function uses the full CML geometry and makes the gauge geometry
        appear as a rain gauge.
        """
        self.update_weights_(da_rad, da_cml=da_cml, da_gauge=da_gauge)
        self.update_x0_block_(self.discretization, da_cml=da_cml, da_gauge=da_gauge)

    def adjust(
        self, 
        da_rad, 
        da_cml=None, 
        da_gauge=None, 
        variogram_model="spherical",
        variogram_parameters={"sill": 0.9, "range": 5000, "nugget": 0.1},
        n_closest=8
    ):
        """Adjust radar field for one time step.

        Evaluates if we have enough observations to adjust the radar field.
        Then adjust radar field to observations using a block kriging variant
        of kriging with external drift.

        The function allows for the user to supply transformation,
        backtransformation and variogram functions.

        Parameters
        ----------
        da_rad: xarray.DataArray
            Gridded radar data. Must contain the lon and lat coordinates as
            well as the projected coordinates xs and ys as a meshgrid.
        da_cml: xarray.DataArray
            CML observations. Must contain the lat/lon coordinates for the CML
            (site_0_lon, site_0_lat, site_1_lon, site_1_lat) as well as the
            projected coordinates (site_0_x, site_0_y, site_1_x, site_1_y).
        da_gauge: xarray.DataArray
            Gauge observations. Must contain the coordinates for the rain gauge
            positions (lat, lon) as well as the projected coordinates (x, y).
        variogram_model: str
            Must be a valid variogram type in pykrige.
        variogram_parameters: str
            Must be a valid parameters corresponding to variogram_model.
        n_closest: int
            Number of closest links to use for interpolation

        Returns
        -------
        da_rad_out: xarray.DataArray
            DataArray with the same structure as the ds_rad but with the CML
            adjusted radar field.
        """
        # Update weights and x0 geometry for CML and gauge
        self.update(da_rad, da_cml=da_cml, da_gauge=da_gauge)

        # Evaluate radar at cml and gauge ground positions
        rad, obs, x0 = self.get_rad_obs_x0_(da_rad, da_cml=da_cml, da_gauge=da_gauge)

        # Get index of not-nan obs
        keep = np.where(~np.isnan(obs) & ~np.isnan(rad) & (obs > 0) & (rad > 0))[0]

        # Setup pykrige with variogram parameters provided by user
        ok = pykrige.OrdinaryKriging(
            x0[keep, 1, int(x0.shape[1] / 2)], #x-midpoint coordinate
            x0[keep, 0, int(x0.shape[1] / 2)], #y-midpoint coordinate
            obs[keep],
            variogram_model=variogram_model,
            variogram_parameters=variogram_parameters,
        )
        
        # Construct variogram using pykrige
        def variogram(h):
            return ok.variogram_function(ok.variogram_model_parameters, h)

        # Remove radar time dimension
        rad_field = da_rad.isel(time=0).data

        # Set zero values to nan, these are ignored in ked function
        rad_field[rad_field <= 0] = np.nan

        # do addtitive IDW merging
        adjusted = bk_functions.merge_ked_blockkriging(
            rad_field,
            da_rad.xs.data,
            da_rad.ys.data,
            rad[keep],
            obs[keep],
            x0[keep],
            variogram,
            obs[keep].size - 1 if obs[keep].size <= n_closest else n_closest,
        )

        # Remove negative values
        adjusted[adjusted < 0] = 0

        return xr.DataArray(
            data=[adjusted],
            coords=da_rad.coords,
            dims=da_rad.dims
        )

