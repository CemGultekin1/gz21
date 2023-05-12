#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 17:40:39 2020

@author: arthur
"""
# import mlflow
import xarray as xr
from yaml import Loader, Dumper,load
# from mlflow import MlflowClient
from gz21.paths import COARSE_DATA_PATH,NEW_COARSE_DATA_PATH
# def find_latest_data_run()->dict:
#     experiment_name = "data"
#     runs = mlflow.search_runs(
#             experiment_names=[experiment_name],
#             filter_string = "tags.mlflow.runName = 'full'"
#     )
#     return runs.loc[len(runs)-1].to_dict()

def load_data_from_past():
    data_file = COARSE_DATA_PATH
    xr_dataset = xr.open_zarr(data_file)
    xr_dataset = xr_dataset.rename(
        dict(
            lat = 'yu_ocean',
            lon = 'xu_ocean',
            Su = 'S_x',
            Sv = 'S_y',
            u = 'usurf',
            v = 'vsurf'
        )
    ).drop('Stemp temp interior_wet_mask wet_density'.split()).isel(depth = 0)
    return xr_dataset#.isel(time= [0,1,2])

def load_data_from_past_():
    data_file = NEW_COARSE_DATA_PATH
    xr_dataset = xr.open_zarr(data_file)
    return xr_dataset

# def load_data_from_run(run_id):
#     data_file =  mlflow.artifacts.download_artifacts(run_id = run_id,artifact_path = "forcing.zarr")
#     xr_dataset = xr.open_zarr(data_file)
#     return xr_dataset


# def load_data_from_runs(run_ids):
#     xr_datasets = list()
#     for run_id in run_ids:
#         xr_datasets.append(load_data_from_run(run_id))
#     return xr_datasets


def load_training_datasets(ds: xr.Dataset, config_fname: str):
    results = []
    with open(config_fname) as f:
        try:
            subdomains = load(f,Loader)#yaml.load(f)
        except FileNotFoundError as e:
            raise type(e)('Configuration file of subdomains not found')
        for subdomain in subdomains:
            coords = subdomain[1]
            lats = slice(coords['lat_min'], coords['lat_max'])
            lons = slice(coords['lon_min'], coords['lon_max'])
            results.append(ds.sel(xu_ocean=lons, yu_ocean=lats))
    return results


def cyclize_dataset(ds: xr.Dataset, coord_name: str, nb_points: int):
    """
    Return a cyclic dataset, with nb_points added on each end, along 
    the coordinate specified by coord_name

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to process.
    coord_name : str
        Name of the coordinate along which the data is made cyclic.
    nb_points : int
        Number of points added on each end.

    Returns
    -------
    New extended dataset.

    """
    # TODO make this flexible
    cycle_length = 360.0
    left = ds.roll({coord_name: nb_points}, roll_coords=True)
    right = ds.roll({coord_name: nb_points}, roll_coords=True)
    right = right.isel({coord_name: slice(0, 2 * nb_points)})
    left[coord_name] = xr.concat((left[coord_name][:nb_points] - cycle_length, left[coord_name][nb_points:]), coord_name)
    right[coord_name] = xr.concat((right[coord_name][:nb_points], right[coord_name][nb_points:] + cycle_length), coord_name)
    new_ds = xr.concat((left, right), coord_name)
    return new_ds