# stripped down IDW interpolator from pycomlink v0.3.10
# in this version you have to supply the xgrid and ygrid directly
from __future__ import print_function
from builtins import zip
import numpy as np
from pykrige import OrdinaryKriging
from idw import Invdisttree

class IdwKdtreeInterpolator:
    def __init__(self, nnear=8, p=2, exclude_nan=True, max_distance=None):
        """A k-d tree based IDW interpolator for points to grid"""
        self.nnear = nnear
        self.p = p
        self.exclude_nan = exclude_nan
        self.max_distance = max_distance
        self.x = None
        self.y = None

    def __call__(self, x, y, z, xi, yi):
        """Do IDW interpolation"""

        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)

        if self.exclude_nan:
            not_nan_ix = ~np.isnan(z)
            x = x[not_nan_ix]
            y = y[not_nan_ix]
            z = z[not_nan_ix]
        self.z = z

        if np.array_equal(x, self.x) and np.array_equal(y, self.y):
            # print('Reusing old `Invdisttree`')
            idw = self.idw
        else:
            # print('Building new `Invdistree`')
            idw = Invdisttree(X=list(zip(x, y)))
            self.idw = idw
            self.x = x
            self.y = y

        zi = idw(
            q=list(zip(xi, yi)),
            z=z,
            nnear=self.nnear,
            p=self.p,
            max_distance=self.max_distance,
        )
        return zi