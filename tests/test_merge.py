#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 10:03:34 2024

@author: erlend
"""

from mergeplg import merge
import xarray as xr
import numpy as np
import pandas as pd

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
        "length": ("cml_id", [2 * np.sqrt(2), 2 * np.sqrt(2), 2 * np.sqrt(2)]),
    },
)

def test_calculate_cml_geometry():
    # Test that the CML geometry is correctly esimtated
    y, x = merge.calculate_cml_geometry(
        ds_cmls.isel(cml_id = [1]), 
        disc = 2 # divides the line into two intervals, 3 points
    )[0]
    assert (y == np.array([-1, 0, 1])).all()
    assert (x == np.array([0, 1, 2])).all()
    
    


def test_block_points_to_lengths():
    # Check that the length matrix is correctly estimated
    l = merge.block_points_to_lengths(
        merge.calculate_cml_geometry(
            ds_cmls.isel(cml_id = [0, 1]), 
            disc = 2
    ))
    l0l0 = l[0, 0] # Lengths from link0 to link0
    l0l1 = l[0, 1] # Lengths from link0 to link0
    l1l0 = l[1, 0] # Lengths from link0 to link0
    l1l1 = l[1, 1] # Lengths from link0 to link0
    
    # Length matrix from line1 to lin1 
    assert (l0l0 == np.array([
        [0, np.sqrt(2), 2*np.sqrt(2)],
        [np.sqrt(2), 0, np.sqrt(2)],
        [2*np.sqrt(2), np.sqrt(2), 0],
    ])).all()
    
    # Length matrix from line1 to lin2 
    assert (l0l1 == np.array([
        [1, 1, np.sqrt(2**2 + 1)],
        [np.sqrt(2**2 + 1), 1, 1],
        [np.sqrt(3**2 + 2**2), np.sqrt(2**2 + 1), 1],
    ])).all()

    # Length matrix from line2 to lin1
    assert (l1l0 == np.array([
        [1, np.sqrt(2**2 + 1), np.sqrt(3**2 + 2**2)],
        [1, 1, np.sqrt(2**2 + 1)],
        [np.sqrt(1+2**2), 1, 1],
    ])).all()
    
    # Length matrix from line2 to lin2
    assert (l1l1 == np.array([
        [0, np.sqrt(2), 2*np.sqrt(2)],
        [np.sqrt(2), 0, np.sqrt(2)],
        [2*np.sqrt(2), np.sqrt(2), 0],
    ])).all()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    