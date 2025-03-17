#! /usr/bin/env python
# This code plots seasonal anomalous state parameters
# for each hemisphere.
# Running this code requires the following additional arguments: [{variable of interest}]['N' or 'S'] 

import numpy as np
import numpy.ma as ma
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
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
POLE=sys.argv[2] #extract pole of interest from arguments

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
# COLLECTION1 ='geosgcm_seaice'
# COLLECTION2='geosgcm_surf'
# COLLECTION3 = 'geosgcm_prog'
EXPDIR='/nobackup/hzafar/Ys2s3_base_9'
HOMDIR=EXPDIR
EXPID=EXPDIR.split('/')[-1]
PLOT_PATH = './plots/'
HAD_fileloc = './ObservationData/HADISST2/'
MERRA_fileloc = './transfer/'
S2S_fileloc = './interm_data/'

# Define misc variables
pngname = EXPID + '_' + statevar +'_'+POLE 
transform = crs=ccrs.PlateCarree()


# Import statevar .nc file
fname = S2S_fileloc + 'preprocessed_' + statevar + '.nc'
ds_s2s = xr.open_dataset(fname)[statevar+'_'+POLE]
# print(ds_s2s)

# define climatologies and plotting variables
if POLE == 'N': 
  SEASON=['M03', 'M09']
  MONTH=['MAR', 'SEP']
  proj = ccrs.NorthPolarStereo()
  
else:
  SEASON=['M09', 'M02']
  MONTH=['SEP', 'FEB']
  proj = ccrs.SouthPolarStereo()
  # Import ice shelves
  ice_shelves = cfeature.NaturalEarthFeature(
        category='physical',
        name='antarctic_ice_shelves_polys',
        scale='10m')

# Create figures
fig, ax = plt.subplots(2, figsize=(10, 16), subplot_kw={'projection': proj})

for i, (sea,mon_name) in enumerate(zip(SEASON,MONTH)):

  if (POLE=='N'):
    ax[i].set_extent([-180, 180, 45, 90], crs=ccrs.PlateCarree())
    ax[i].add_feature(cfeature.LAND, facecolor='lightgrey',zorder=1) 
  else:
    ax[i].set_extent([-180, 180, -45, -90], crs=ccrs.PlateCarree())
    ax[i].add_feature(cfeature.LAND, facecolor='lightgrey', zorder=1)
    #ax.add_feature(cfeature.COASTLINE,zorder=1)
    ax[i].add_feature(ice_shelves, facecolor='lightgrey',zorder=1)

  gl = ax[i].gridlines(crs=ccrs.PlateCarree(), draw_labels=True, y_inline=False,
          xlocs=range(-180,181,30), ylocs=range(-180,181,30), rotate_labels=False,
          linewidth=0.5, color='white', alpha=0.5)
  gl.right_labels=False

  mon = int(sea[-2:])
  ds_s2s_selmon = ds_s2s.sel(time=ds_s2s.time.dt.month == mon)
  #Convert from kg/m2-s to mm/day
  ds_s2s_selmon = ds_s2s_selmon*86400
  
  #Take mean
  ds_s2s_selmon = ds_s2s_selmon.mean(dim='time')
  ds_s2s_selmon=ds_s2s_selmon.where(ds_s2s_selmon!=0)

  # print(ds_s2s_selmon)
  plot = ds_s2s_selmon.plot(ax=ax[i],transform=transform,cmap='Blues',add_colorbar=False)
  ax[i].set_title(mon_name,fontsize=18)

cbar = fig.colorbar(plot, ax=ax.ravel().tolist())
cbar.set_label('Snowfall (mm/day)')

fig.savefig(PLOT_PATH+pngname)