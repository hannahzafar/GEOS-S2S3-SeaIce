#! /usr/bin/env python
# This code analyzes and plots seasonal sic for anomalous state parameters
# for the selected hemisphere.
# Running this code requires the following additional arguments: [{variable of interest}]['N' or 'S'] 

import numpy as np
import numpy.ma as ma
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
mpl.use('Agg')
import pandas as pd
import xarray as xr
xr.set_options(display_width=95, display_max_rows=40, display_expand_attrs=False)
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
scenarios = ['High', 'Low', 'Mean']
pngname = EXPID+'_aice_' + statevar +'scenarios_'+POLE

# Define climatology and plotting settings for N and S poles
transform = crs=ccrs.PlateCarree()
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

# colormap and levels
cmap = plt.cm.RdBu_r
levels = np.array([-0.8, -0.6, -0.4, -0.2, -0.1, 0.1, 0.2, 
                        0.4, 0.6, 0.8])

# Generate figure/axes and set parameters
# fig, axes = plt.subplots(2,3, figsize=(16, 10), constrained_layout=True, subplot_kw={'projection': proj},gridspec_kw={'hspace': None, 'wspace': None})
fig, axes = plt.subplots(2, 3, figsize=(16, 12), constrained_layout=True,
                         subplot_kw={'projection': proj},
                         gridspec_kw={#'width_ratios': [1, 1, 0.95],
                                      'hspace': 0.01, 'wspace': 0.06})

fig.suptitle(f'{statevar_name} Effects on Sic',fontsize=12)

for i, ax in enumerate(fig.axes):
  if (POLE=='N'):
    ax.set_extent([-180, 180, 45, 90], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='darkgrey', zorder=2) 
  else:
    ax.set_extent([-180, 180, -45, -90], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='darkgrey', zorder=2)
    #ax.add_feature(cfeature.COASTLINE,zorder=1)
    ax.add_feature(ice_shelves, facecolor='darkgrey',zorder=2)
  
  gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, y_inline=False,
          xlocs=range(-180,181,30), ylocs=(-60,60), ylim=(-60,60), rotate_labels=False,
          linewidth=1, color='dimgrey', alpha=0.5)
          
  gl.xlabel_style = {'size': 12}
  gl.ylabel_style = {'size': 12}
  gl.left_labels=False
  
  # fix for cartopy draw error:
  plt.draw()
  for ea in gl.right_label_artists:
      ea.set_visible(True)

# Import preprocessed scenarios data
fname = S2S_fileloc + 'preprocessed_aice_' + statevar + 'scenarios_seas_' + POLE + '.nc'
ds_scenarios = xr.open_dataset(fname)

for i, (sea,mon_name) in enumerate(zip(SEASON,MONTH)):
  
  # Select month data
  mon = int(sea[-2:])
  ds_scenarios_sel = ds_scenarios.sel(month = mon)
  
  for j, scenario in enumerate(scenarios):
    if scenario == 'High':
      Diff = ds_scenarios_sel['High_AICE']-ds_scenarios_sel['High_sic']
    if scenario == 'Low':
      Diff = ds_scenarios_sel['Low_AICE']-ds_scenarios_sel['Low_sic']
    if scenario == 'Mean':
      Diff = ds_scenarios_sel['Mean_AICE']-ds_scenarios_sel['Mean_sic']
    
    # Drop zeroes
    # Diff = Diff.where(Diff!=0)

    # plot = Diff.plot(ax=axes[i,j],transform=transform,cmap=cmap, vmin=-0.9, vmax=0.9, add_colorbar=False)
    plot = Diff.plot.pcolormesh(ax=axes[i,j],transform=transform,cmap=cmap,levels=levels,vmin=-0.8,vmax=0.8,extend='both',add_colorbar=False,zorder=0)

 
    # Plot settingsxw
    axes[i,0].set_yticks([])
    axes[i,0].set_ylabel(mon_name,fontsize=20, rotation='horizontal',labelpad=40,weight='bold')
    axes[i,j].set_title('')
    axes[0,j].set_title(scenario,fontsize=22)


cbar = fig.colorbar(plot, ax=axes.ravel().tolist(), orientation='horizontal',shrink=0.6,ticks=levels, extend='both')
cbar.ax.tick_params(labelsize=18)
cbar.set_label('GEOS-S2S-3 — HadISST2', size=20)
minorticks = [0]
cbar.set_ticks(minorticks, minor=True)
fig.savefig(PLOT_PATH+pngname, dpi=fig.dpi, bbox_inches='tight')
# plt.show()