#!/usr/bin/env python
# High/Low Difference monthly averages
# This code analyzes and plots monthly sic for anomalous state parameters
# for the selected hemisphere compared to anomalous HadISST.
# Running this code requires the following additional arguments: [{variable of interest}]['N' or 'S'] 

import numpy as np
import numpy.ma as ma
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pds
import xarray as xr
# xr.set_options(display_width=95, display_max_rows=40, display_expand_attrs=False)
import cartopy
import warnings
warnings.filterwarnings('ignore')
from cartopy import crs as ccrs, feature as cfeature  
import glob
import time
import datetime
import sys
import os
# import re
# import struct

# Extract variables from arguments:
VAR = sys.argv[1]
POLE= sys.argv[2]

# Variable of interest:
if VAR == 'SLP':
  statevar = 'SLP'
  statevar2 = statevar #MERRA2 name
  statevar_name = 'Sea Level Pressure'

if VAR == 'SNO':
  statevar = 'SNO'
  statevar2 = 'PRECSNO' #MERRA2 name
  statevar_name = 'Snowfall'

if VAR == 'TS':
  statevar = 'TS'
  statevar2 = 'TS' #MERRA2 name
  statevar_name = 'Skin Temperature'

# Define file locations
COLLECTION1 ='geosgcm_seaice'
COLLECTION2='geosgcm_surf'
COLLECTION3 = 'geosgcm_prog'
EXPDIR='/nobackup/hzafar/Ys2s3_base_9'
HOMDIR=EXPDIR
EXPID=EXPDIR.split('/')[-1]
PLOT_PATH = './plots/'
HAD_fileloc = './ObservationData/HADISST2/'
MERRA_fileloc = './transfer/'
S2S_fileloc = './interm_data/'

# Define scenarios
scenarios = ['High', 'Low', 'Control']
pngname = EXPID+'_aice_' + statevar +'3_'+POLE

# Define misc variables
months = np.arange(1,13)
monthlabel = [datetime.date(1900, mon, 1).strftime('%b') for mon in months]

# Create figure/axes
fig, axes = plt.subplots(2,1, figsize=(8,6),constrained_layout=True)

# Import preprocessed data from 3A
fname = S2S_fileloc + 'preprocessed_aice_diff_' + statevar + '_' +POLE + '.nc'
ds_seaice = xr.open_dataset(fname)

# Plot ds_seaice to check
  # transform = crs=ccrs.PlateCarree()
  # if POLE == 'N': 
  #   proj = ccrs.NorthPolarStereo()
  # else:
  #   proj = ccrs.SouthPolarStereo()

  # fig, axes = plt.subplots(3,2, figsize=(6, 10), subplot_kw={'projection': proj})
  # for i, ax in enumerate(fig.axes):
  #   if (POLE=='N'):
  #     ax.set_extent([-180, 180, 45, 90], crs=ccrs.PlateCarree())
  # #     # ax.add_feature(cfeature.LAND, facecolor='lightgrey',zorder=1) 
  # #   else:
  # #     ax.set_extent([-180, 180, -45, -90], crs=ccrs.PlateCarree())
  # #     # ax.add_feature(cfeature.LAND, facecolor='lightgrey', zorder=1)

  # ds_seaice_sel = ds_seaice.sel(month=1)
  # # ds_seaice_sel.Control_AICE.plot(ax=axes[0],transform=transform)
  # # ds_seaice_sel.Control_FRSEAICE.plot(ax=axes[1],transform=transform)
  # # ds_seaice_sel.Control_sic.plot(ax=axes[2],transform=transform)
  # ds_seaice_sel.High_FROCEAN.plot(ax=axes[0,0],transform=transform)
  # ds_seaice_sel.Low_FROCEAN.plot(ax=axes[1,0],transform=transform)
  # ds_seaice_sel.Control_FROCEAN.plot(ax=axes[2,0],transform=transform)
  # High_diff = ds_seaice_sel.High_FROCEAN - ds_seaice_sel.Low_FROCEAN
  # Low_diff = ds_seaice_sel.Low_FROCEAN - ds_seaice_sel.Control_FROCEAN
  # Control_diff = ds_seaice_sel.Control_FROCEAN - ds_seaice_sel.High_FROCEAN
  # High_diff.plot(ax=axes[0,1],transform=transform)
  # Low_diff.plot(ax=axes[1,1],transform=transform)
  # Control_diff.plot(ax=axes[2,1],transform=transform)
  # fig.savefig(PLOT_PATH+'FROCEAN_compare')
  # sys.exit()

# Compute high/low S2S and Had differences
Diff = abs(ds_seaice['High_AICE']-ds_seaice['Low_AICE']) - abs(
      ds_seaice['High_sic']-ds_seaice['Low_sic'])

Diff = Diff*ds_seaice.High_FROCEAN
# print(Diff)

# Drop zeros
Diff = Diff.where(Diff!=0)

# Calculate area of each grid cell
R_earth = 6357 #6378 #km
Area = (np.pi/180)*(R_earth**2)*np.abs(
  (np.sin(np.deg2rad(Diff.lat+0.5))
  -np.sin(np.deg2rad(Diff.lat-0.5))))*(Diff.lon+0.5-(Diff.lon-0.5)) #Diff in lon is 1 deg
# print(Area)

Diff_weighted = Diff*Area
# print(Diff_weighted)

# Plot Diff-weighted to check
  # transform = crs=ccrs.PlateCarree()
  # if POLE == 'N': 
  #   proj = ccrs.NorthPolarStereo()
    
  # else:
  #   proj = ccrs.SouthPolarStereo()

  # fig, axes = plt.subplots(1,1, figsize=(6, 10), subplot_kw={'projection': proj})

  # for i, ax in enumerate(fig.axes):
  #   if (POLE=='N'):
  #     ax.set_extent([-180, 180, 45, 90], crs=ccrs.PlateCarree())
  #     ax.add_feature(cfeature.LAND, facecolor='lightgrey',zorder=1) 
  #   else:
  #     ax.set_extent([-180, 180, -45, -90], crs=ccrs.PlateCarree())
  #     ax.add_feature(cfeature.LAND, facecolor='lightgrey', zorder=1)

  # Diff_weighted_sel = Diff_weighted.sel(month = 1)

  # Diff_weighted_sel.plot(ax=axes,transform=transform)
  # plt.show()

# Calculate mean, std dev, sum squares
Mean = Diff_weighted.mean(dim=["lat","lon"],skipna=True)
Std_dev = Diff_weighted.std(dim=["lat","lon"],skipna=True)
Diff_square = (Diff_weighted)**2
RMSE = Diff_square.mean(dim=["lat","lon"],skipna=True)
# print(RMSE)
# sys.exit()

# Plot mean + variance
axes[0].errorbar(months, Mean.values, Std_dev.values,c='lightblue',ls='none')
axes[0].plot(months,Mean.values,c='b')
axes[0].set_title('Mean Difference')

axes[1].plot(months, RMSE.values,c='tab:orange')
axes[1].set_title('RMSE')

[axes[n].set_xticks(months) for n in range(2)]
axes[0].set_xticklabels('')
axes[1].set_xticklabels(monthlabel)
fig.suptitle(f'Anomalous {statevar_name} Difference',fontsize=15)
fig.suptitle(f'{POLE}H Anomalous {statevar_name} \n $S2S_(High - Low)$- $Had_(High - Low)$',fontsize=15)
# plt.show()

fig.savefig(PLOT_PATH+pngname)
# fig.savefig(PLOT_PATH+pngname, dpi=fig.dpi, bbox_inches='tight')

