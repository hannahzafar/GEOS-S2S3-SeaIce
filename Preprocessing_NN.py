#! /usr/bin/env python
# This code computes data for export for HZ's neural network project

import numpy as np
# np.set_printoptions(threshold=np.inf) 
import numpy.ma as ma
import xarray as xr
# xr.set_options(display_width=95, display_max_rows=40, display_expand_attrs=False)
import pandas as pd
# pd.set_option("display.max_rows", None)
# pd.set_option("display.max_columns", None)
# pd.set_option('display.max_seq_items',None)
# pd.options.mode.chained_assignment = None  # default='warn'
import dask 
dask.config.set({"array.slicing.split_large_chunks": True}) 
import glob
import datetime as dt
import time
import sys
import os

# Define file locations
COLLECTION1 ='geosgcm_seaice'
COLLECTION2='geosgcm_surf'
COLLECTION3 = 'geosgcm_prog'
HAD_fileloc = './ObservationData/HADISST2/'
MERRA_fileloc = './transfer/'
S2S_fileloc = './interm_data/'

#Import already preprocessed data
## Extent
fname = S2S_fileloc + 'preprocessed_seaice_extent.csv'
var = 'Extent_N'
ds_extent = pd.read_csv(fname,index_col=0, parse_dates=True)[var]
ds_extent = ds_extent*1e-6
ds_extent.index = ds_extent.index.normalize()
# print(ds_extent)

## HadISST2 data
fname= HAD_fileloc + 'HadISST.2.2.1.0_NH_sea_ice_extent.txt'
ds_Had = pd.read_csv(fname,delim_whitespace=True)
ds_Had.columns = ['Year', 'Month', 'Had_Extent']
# ds_Had['datetime']=pd.to_datetime(ds_Had.assign(Day=1)[['Year','Month','Day']])
startyr = 1990
endyr = 2010
ds_Had_years = ds_Had[ds_Had['Year'].between(startyr, endyr)]
ds_Had_monthlyavg = ds_Had_years.groupby('Month')['Had_Extent'].mean()
# print(ds_Had_monthlyavg)

## Compute Extent - average monthly Had
ds_extent = ds_extent.to_frame()
ds_extent = ds_extent.join(ds_Had_monthlyavg, on=ds_extent.index.month)
ds_extent_diff = ds_extent.Extent_N - ds_extent.Had_Extent
ds_extent_diff.name = 'Extent_Had_Diff_N'
# print(ds_extent_diff)

## SLP
fname = S2S_fileloc + 'preprocessed_SLP.csv'
var = 'SLP_N'
ds_SLP = pd.read_csv(fname,index_col=0, parse_dates=True)[var]
# print(ds_SLP)

## SNO
fname = S2S_fileloc + 'preprocessed_SNO.csv'
var = 'SNO_N'
ds_SNO = pd.read_csv(fname,index_col=0, parse_dates=True)[var]
ds_SNO.index = ds_SNO.index.normalize()
# print(ds_SNO)

# TS
fname = S2S_fileloc + 'preprocessed_TS.csv'
var = 'TS_N'
ds_TS = pd.read_csv(fname,index_col=0, parse_dates=True)[var]
# print(ds_TS)

# Combine dataframes
frames = [ds_SLP, ds_SNO, ds_TS, ds_extent_diff]
combine = pd.concat(frames,axis=1)
combine = combine[combine.index.year>2019]
combine.to_csv('interm_data/preprocessed_NN_data.csv')