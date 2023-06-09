#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 15:19:16 2020

@author: arthur
This script computes the subgrid forcing for a given region, using
data from cmip2.6 on one of the pangeo data catalogs.
Command line parameters include region specification.
Run cmip26 -h to display help.
"""
from os.path import join
import os

import argparse
from gz21.paths import TEMP
import xarray as xr
from dask.diagnostics import ProgressBar
import mlflow

from data.utils import cyclize_dataset
from data.coarse import eddy_forcing
from data.pangeo_catalog import get_patch_from_file
import logging
import tempfile
def main():
    # logging_level = os.environ.get('LOGGING_LEVEL')
    
    # if logging_level is not None:
    #     logging_level = getattr(logging, logging_level)
    #     logging.basicConfig(level=logging_level)
    logger = logging.getLogger(__name__)
    # logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    


    DESCRIPTION = 'Read data from the CM2.6 and \
            apply coarse graining. Stores the resulting dataset into an MLFLOW \
            experiment within a specific run.'

    # data_location = tempfile.mkdtemp(dir='/scratch/ag7531/temp/')

    # Parse the command-line parameters
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('bounds', type=float, nargs=4, help='min lat, max_lat,\
                        min_long, max_long')
    parser.add_argument('--global_', type=int, help='True if global data. In this\
                        case the data is made cyclic along longitude', 
                        default=1)
    parser.add_argument('--ntimes', type=int, default=10000, help='number of days,\
                        starting from first day.')
    parser.add_argument('--CO2', type=int, default=0, choices=[0, 1], help='CO2\
                        level, O (control) or 1 (1 percent CO2 increase)')
    parser.add_argument('--factor', type=int, default=0,
                        help='Factor of degrading. Should be integer > 1.')
    parser.add_argument('--chunk_size', type=str, default='50',
                        help='Chunk size along the time dimension')
    parser.add_argument('--num_cpus', type=str, default='2',
                        help='Number of cpus used.')
    params = parser.parse_args()


    # Retrieve the patch of data specified in the command-line args
    patch_data, grid_data = get_patch_from_file(params.ntimes, params.bounds,
                                    params.CO2, 'usurf', 'vsurf')
    logger.debug(patch_data)
    logger.debug(grid_data)

    # If global data, we make the dataset cyclic along longitude
    if params.global_ == 1:
        logger.info('Cyclic data... Making the dataset cyclic along longitude...')
        # patch_data = cyclize_dataset(patch_data, 'xu_ocean', params.factor)
        # grid_data = cyclize_dataset(grid_data, 'xu_ocean', params.factor)
        # Rechunk along the cyclized dimension
        print(f'patch_data  = patch_data.chunk(chunks = dict(time = ({int(params.num_cpus)}),xu_ocean = (-1)))')
        patch_data  = patch_data.chunk(chunks = dict(time = (int(params.num_cpus)),xu_ocean = (-1)))
        # grid_data = grid_data.chunk(xu_oc

    logger.debug('Getting grid data locally')
    # grid data is saved locally, no need for dask
    grid_data = grid_data.compute()

    logger.debug('Mapping blocks')
    # Calculate eddy-forcing dataset for that particular patch
    debug_mode = os.environ.get('DEBUG_MODE')
    if params.factor != 0 and not debug_mode:
        scale_m = params.factor
        def func(block):
            return eddy_forcing(block, grid_data, scale=scale_m)
        def grouped_func(block):
            return eddy_forcing(block.groupby("time"), grid_data, scale=scale_m)
        
        template = patch_data.coarsen(dict(xu_ocean=int(scale_m),
                                        yu_ocean=int(scale_m)),
                                    boundary='trim').mean()
        template2 = template.copy()
        template2 = template2.rename(dict(usurf='S_x', vsurf='S_y'))
        template = xr.merge((template, template2))
        
        forcing = xr.map_blocks(grouped_func, patch_data, template=template,)
        forcing = forcing.chunk(dict(time = (-1)))
    elif not debug_mode:
        scale_m = params.scale * 1e3
        forcing = eddy_forcing(patch_data, grid_data, scale=scale_m, method='mean')
    else:
        logger.info('!!!Debug mode!!!')
        forcing = patch_data

    # Progress bar
    ProgressBar().register()

    # Specify input vs output type for each variable of the dataset. Might
    # be used later on for training or testing.
    if not debug_mode:
        forcing['S_x'].attrs['type'] = 'output'
        forcing['S_y'].attrs['type'] = 'output'
        forcing['usurf'].attrs['type'] = 'input'
        forcing['vsurf'].attrs['type'] = 'input'

    # Crop according to bounds
    bounds = params.bounds
    forcing = forcing.sel(xu_ocean=slice(bounds[2], bounds[3]),
                        yu_ocean=slice(bounds[0], bounds[1]))

    logger.info('Preparing forcing data')
    logger.debug(forcing)
    # export data
    
    tmpdirname = tempfile.mkdtemp(dir=TEMP)
    # with tempfile.TemporaryDirectory() as tmpdirname:
    print(f"data_location = {tmpdirname}")
    forcing.to_zarr(join(tmpdirname, 'forcing.zarr'), mode='w')
    # Log as an artifact the forcing data
    logger.info('Logging processed dataset as an artifact...')
    # TODO this part increases the memory requirement and causing a out-of-memory exit
    # Currently handled by moving the folder manually into mlruns
    mlflow.log_artifact(join(tmpdirname, 'forcing.zarr'))
    logger.info('Completed...')
    
    
if __name__ == '__main__':
    main()