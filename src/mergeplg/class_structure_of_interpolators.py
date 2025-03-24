# class structure of interpoltors for point to grid and line to grid
# usable with IDW, Ordinary Kriging, Block Kriging and KED.

from abc import abstractmethod

import numpy as np
import xarray as xr


class PointsToGridInterpolator:
    def __init__(self, ds_points, ds_grid):
        self._interpolator = self._init_interpolator(ds_points, ds_grid)
        self.ds_points = (
            ds_points  # TODO: better to just keep a trimmed down version without vars?
        )
        self.ds_grid = (
            ds_grid  # TODO: better to just keep a trimmed down version without vars?
        )
        # self.x = ds_points.x
        # self.y = ds_points.y
        # self.x_grid = ds_grid.xs
        # self.y_grid = ds_grid.ys

    def __call__(self, da_points):
        self._interpolator = self._maybe_update_interpolator(da_points)
        return self._interpolator(da_points)

    @abstractmethod
    def _init_interpolator(self, ds_points, ds_grid):
        # needs to return the interpolator
        raise NotImplementedError()

    def _maybe_update_interpolator(self, da_points):
        if not np.array_equal(da_points.x, self.x) or not np.array_equal(
            da_points.y, self.y
        ):
            return self._init_interpolator(da_points, self.ds_grid)
        else:
            return self._interpolator


class PointsToGridInterpolatorIDW(PointsToGridInterpolator):
    def __init__(
        self,
        ds_points,
        ds_grid,
        nnear=8,
        p=2,
        exclude_nan=True,
        max_distance=60e3,
    ):
        self.ds_points = ds_points
        self.ds_grid = ds_grid

        self._interpolator = self._init_interpolator(
            ds_points=ds_points, ds_grid=ds_grid
        )

        # not sure if this is good/okay to keep these args here to be used in self._interpolator
        self.nnear = nnear
        self.p = p
        self.exclude_nan = exclude_nan
        self.max_distance = max_distance

        self.xy_radar = np.array(
            list(
                zip(ds_grid.xs.data.flatten(), ds_grid.ys.data.flatten(), strict=False)
            )
        )

    def _init_interpolator(self, ds_points, ds_grid):
        from mergeplg.radolan.idw import Invdisttree

        self.x = ds_points.x
        self.y = ds_points.y
        return Invdisttree(X=np.array(list(zip(ds_points.x.data, ds_points.y.data))))

    def __call__(self, da_points):
        self._interpolator = self._maybe_update_interpolator(da_points)
        zi = self._interpolator(
            q=self.xy_radar,
            z=da_points.data,
            nnear=self.nnear,
            p=self.p,
            max_distance=self.max_distance,
        )
        print(self.xy_radar)
        return xr.DataArray(
            dims=["y", "x"],
            coords={
                "y": self.ds_grid.y,
                "x": self.ds_grid.x,
                "ys": self.ds_grid.ys,
                "xs": self.ds_grid.xs,
            },
            data=zi.reshape(self.ds_grid.xs.data.shape),
        )


class LinesToGridInterpolator:
    def __init__(self, x1, y1, x2, y2, x_grid, y_grid):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.x_grid = x_grid
        self.y_grid = y_grid
        self._init_interpolator(x1, y1, x2, y2, x_grid, y_grid)

    def __call__(self, x1, y1, x2, y2, z):
        self._maybe_update_interpolator(x1, y1, x2, y2)
        return self._interpolator(x1, y1, x2, y2, z)

    @abstractmethod
    def _init_interpolator(self, x1, y1, x2, y2, z):
        # here you have to define `self._interpolator`
        raise NotImplementedError()

    def _maybe_update_interpolator(self, x1, y1, x2, y2):
        if (
            not np.array_equal(self.x1, x1)
            or not np.array_equal(self.y1, y1)
            or not np.array_equal(self.x2, x2)
            or not np.array_equal(self.y2, y2)
        ):
            self._init_interpolator(x1, y1, x2, y2, self.x_grid, self.y_grid)


class LinesToGridBlockKriging(LinesToGridInterpolator):
    # and so on...
    pass


class MergeBase:
    def __init__(self, ds_grid, ds_points=None, ds_lines=None):
        self.ds_grid = ds_grid
        self.ds_points = ds_points
        self.ds_lines = ds_lines

    @abstractmethod
    def __call__(self, da_grid, da_points=None, da_lines=None):
        raise NotImplementedError()


class MergeIDW(MergeBase):
    def __init__(
        self, ds_grid, ds_points=None, ds_lines=None, max_distance=10e3, min_points=5
    ):
        super().__init__(ds_grid, ds_points, ds_lines)
        idw_interpolator = PointsToGridInterpolatorIDW()
