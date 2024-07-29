#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 14:22:46 2024

@author: erlend
"""

import numpy as np
import pandas as pd
import pykrige

from .interpolator import IdwKdtreeInterpolator

def block_points_to_lengths(x0):
    # Distances between all discretization points 
    delta_x = np.array([x0[i][1] - x0[j][1].reshape(
        -1, 1) for i in range(x0.shape[0]) for j in range(x0.shape[0])])
    delta_y = np.array([x0[i][0] - x0[j][0].reshape(
        -1, 1) for i in range(x0.shape[0]) for j in range(x0.shape[0])])
    
    lengths_point_l = np.sqrt(delta_x**2 + delta_y**2)
    
    # Shape = n_lines x n_lines x disc x disc
    return lengths_point_l.reshape(
        int(np.sqrt(lengths_point_l.shape[0])), 
        int(np.sqrt(lengths_point_l.shape[0])),
        lengths_point_l.shape[1],
        lengths_point_l.shape[2],
    )

def calculate_cml_geometry(ds_cmls, disc=8):    
    # Calculate discretized possitions along the lines, store in numy array
    xpos = np.zeros([ds_cmls.cml_id.size, disc+1]) # shape (line, possition)
    ypos = np.zeros([ds_cmls.cml_id.size, disc+1])
    for block_i, cml_id in enumerate(ds_cmls.cml_id):   
        x_a = ds_cmls.sel(cml_id = cml_id).site_0_x.values
        y_a = ds_cmls.sel(cml_id = cml_id).site_0_y.values
        x_b = ds_cmls.sel(cml_id = cml_id).site_1_x.values
        y_b = ds_cmls.sel(cml_id = cml_id).site_1_y.values
        
        # for all dicretization steps in link estimate its place on the grid
        for i in range(disc+1): 
            xpos[block_i, i] = x_a + (i/disc)*(x_b - x_a) 
            ypos[block_i, i] = y_a + (i/disc)*(y_b - y_a) 
            
    # Store CML possitions in common array
    return np.array([ypos, xpos]).transpose([1, 0, 2])    

def estimate_kriging_parameters(
        ds_cmls, 
        T = 0, 
        hr_min=None, 
        hr_max=None, 
        drop_zero = True,
        min_obs = 5,
    ):
    # Estiamte CML midpoint
    x_ = ((ds_cmls.site_0_x + ds_cmls.site_1_x)/2).data 
    y_ = ((ds_cmls.site_0_y + ds_cmls.site_1_y)/2).data
    param_kriging = {}
            
    # Timestep to estimate parameters for variogram and transformation
    for time_mid in range(ds_cmls.time.size): 
        x, y, values = [], [], []
        
        # Set lower and upper timesteps
        lower = time_mid - T
        if lower < 0:
            lower = 0
        upper = time_mid + T + 1
        if upper > ds_cmls.time.size:
            upper = ds_cmls.time.size
        
        # Timesteps in window
        for time in ds_cmls.time[lower:upper]:
            val_ = ds_cmls.sel(time = time).data        
            if drop_zero:
                keep = (val_> 0) 
            else:
                keep = np.ones(val_.shape).astype(bool)
            values.append(val_[keep])
            x.append(x_[keep])
            y.append(y_[keep])
            
        x = np.concatenate(x)
        y = np.concatenate(y)
        values = np.concatenate(values)
        
        # Estimate variogram if enough observations
        if x.size > min_obs:           
            try: 
                ok = pykrige.OrdinaryKriging(
                    x, 
                    y, 
                    values, 
                    variogram_model='exponential',
                )
                sill, hr, nugget = ok.variogram_model_parameters
                
                # Adjust hr to bounds
                if hr_min is not None:
                    if hr < hr_min:
                        hr = hr_min
                if hr_max is not None:
                    if hr > hr_max:
                        hr = hr_max
                        
                param_kriging[ds_cmls.time.data[time_mid]] = [sill, hr, nugget]
            except:
                param_kriging[ds_cmls.time.data[time_mid]] = [np.nan, np.nan, np.nan]
                
            
        else:
            param_kriging[ds_cmls.time.data[time_mid]] = [np.nan, np.nan, np.nan]
    
    return pd.DataFrame.from_dict(
        param_kriging, 
        orient = 'index', 
        columns=['sill', 'hr', 'nugget']
    )


def merge_additive_IDW(ds_cmls, ds_rad, adjust_where_radar = True, min_obs=5):
    X, Y = ds_rad.xs.data, ds_rad.ys.data
    
    # Array for storing interpolated values
    shift = np.zeros(X.shape)
        
    # Select time step
    cml_obs = ds_cmls.R_diff.data #
    rad_field = ds_rad.rainfall_amount.data 
            
    # Do interpolation where there is radar observations and CML
    # currently this is regulated using nan in the time series, not optimal I guess
    if adjust_where_radar:
        keep = ~np.isnan(cml_obs)
    else:
        keep = np.ones(cml_obs.shape).astype(bool)
        
    # Select the CMLs to keep 
    cml_i_keep = np.where(keep)[0] 
    cml_obs = cml_obs[cml_i_keep]
    
    if cml_i_keep.size > min_obs: # enough observations
        x = ds_cmls.isel(cml_id = cml_i_keep).x.data
        y = ds_cmls.isel(cml_id = cml_i_keep).y.data
        z = cml_obs
        

        # Gridpoints to interpolate (drawback: large arrays with nan if you only want to interpolate at gauge points)
        mask = np.isnan(rad_field) | (rad_field == 0)
        xgrid = X[~mask]
        ygrid = Y[~mask]            
        
        if xgrid.size > 0:
            #IDW interpolator kdtree (from pycomlink)
            idw_interpolator = IdwKdtreeInterpolator(
                nnear=8, 
                p=2, # 
                exclude_nan=True, 
            )
            estimate = idw_interpolator(x=x, y=y, z=z, xi=xgrid, yi=ygrid)
            shift[~mask] = estimate
            
    # create xarray object similar to ds_rad
    ds_rad_out = ds_rad[['rainfall_amount']].copy()
    
    # Store shift data
    ds_rad_out['shift'] = (('y', 'x'), shift)
    
    # Apply shift where we have radar observations
    if adjust_where_radar:
        ds_rad_out['shift'] = ds_rad_out['shift'].where(ds_rad_out.rainfall_amount > 0, 0)
        
    # Adjust field
    ds_rad_out['adjusted'] = (('y', 'x'), ds_rad_out['shift'].data  + ds_rad_out.rainfall_amount.data)
    
    # Set negative values to zero
    ds_rad_out['adjusted'] = ds_rad_out.adjusted.where(ds_rad_out.adjusted > 0, 0)

    # Return dataset with adjusted values
    return ds_rad_out.adjusted


def merge_additive_BlockKriging(
        ds_cmls, 
        ds_rad, 
        x0, 
        nugget=0, 
        C0=1, 
        hr=1, 
        adjust_where_radar = True, 
        min_obs = 5,
    ):
    # Grid coordinates
    X, Y = ds_rad.xs.data, ds_rad.ys.data
    
    # Array for storing interpolated values
    shift = np.zeros(X.shape)
        
    # To numpy for fast lookup
    diff = ds_cmls.R_diff.data 
    rad_field = ds_rad.rainfall_amount.data 
    
    # Do interpolation where there is radar observations and CML
    # currently this is regulated using nan in the time series, not optimal I guess
    if adjust_where_radar:
        keep = ~np.isnan(diff)
    else:
        keep = np.ones(diff.shape).astype(bool)
        
    # Select the CMLs to keep 
    cml_i_keep = np.where(keep)[0] 
    diff = diff[cml_i_keep]
    
    # Adjust radar if enough observations and hr is not nan
    if ~np.isnan(hr) & (cml_i_keep.size > min_obs):
        
        # Define varigram
        def variogram(h):
            return nugget + (C0 - nugget)*(1 - np.exp(-h*3/(hr)))
        
        # Length between all CML
        lengths_point_l = block_points_to_lengths(x0)
       
        # estimate mean variogram over link geometries 
        cov_block = variogram(lengths_point_l[cml_i_keep, :][:, cml_i_keep]).mean(axis=(2, 3))

        # Create Kriging matrix               
        mat = np.zeros([cov_block.shape[0] + 1, cov_block.shape[1] + 1])
        mat[:cov_block.shape[0], :cov_block.shape[1]] = cov_block
        mat[-1, :-1] = np.ones(cov_block.shape[1]) # non-bias condition
        mat[:-1, -1] = np.ones(cov_block.shape[0]) # lagrange multipliers
            
        # Calc the inverse, only dependent on geometry (and radar for KED)
        A_inv = np.linalg.pinv(mat)
            
        # Skip radar pixels with np.nan
        mask = np.isnan(rad_field) 
        
        # Skip radar pixels with zero
        if adjust_where_radar:
            mask = mask | (rad_field == 0)
        
        # Grid to visit
        x, y = X[~mask], Y[~mask]
        
        # array for storing CML-radar merge
        estimate = np.zeros(x.shape)
        
        # Compute the contributions from all CMLs to a point  
        for i in range(x.size):
            # compute target, that is R.H.S of eq 15 (jewel2013)
            delta_x = (x0[cml_i_keep, 1] - x[i])
            delta_y = (x0[cml_i_keep, 0] - y[i])
            lengths = np.sqrt(delta_x**2 + delta_y**2)
            target =  variogram(lengths).mean(axis = 1)
        
            target = np.append(target, 1) # non bias condition
            
            # compuite weigths
            w = (A_inv@target)[:-1]
            
            # its then the sum of the CML values (eq 8, see paragraph after eq 15)
            estimate[i] = diff@w
            
        shift[~mask] = estimate
        
    # create xarray object similar to ds_rad
    ds_rad_out = ds_rad[['rainfall_amount']].copy()
    
    # Store shift data
    ds_rad_out['shift'] = (('y', 'x'), shift)
    
    # Apply shift where we have radar observations
    if adjust_where_radar:
        ds_rad_out['shift'] = ds_rad_out['shift'].where(ds_rad_out.rainfall_amount > 0, 0)
    
    # Adjust field
    ds_rad_out['adjusted'] = (('y', 'x'), ds_rad_out['shift'].data  + ds_rad_out.rainfall_amount.data)
    
    # Set negative values to zero
    ds_rad_out['adjusted'] = ds_rad_out.adjusted.where(ds_rad_out.adjusted > 0, 0)

    # Return dataset with adjusted values
    return ds_rad_out.adjusted