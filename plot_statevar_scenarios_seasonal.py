#! /usr/bin/env python
# This code plots seasonal anomalous state parameters
# for each hemisphere.
# Running this code requires the following additional arguments: [{variable of interest}]['N' or 'S'] 

import numpy as np
import numpy.ma as ma
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
mpl.use('Agg')
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
scenarios = ['High', 'Low', 'Mean']
pngname = EXPID + '_' + statevar +'_scenarios_'+POLE 
transform = crs=ccrs.PlateCarree()
# Blues_cmap = plt.cm.get_cmap('Blues', 256)
# colors = Blues_cmap(np.linspace(0.5, 1, 256))
# custom_cmap = LinearSegmentedColormap.from_list('top_half_blues', colors)

# Import statevar .nc file
fname = S2S_fileloc + 'preprocessed_' + statevar + '.nc'
ds_s2s_state = xr.open_dataset(fname)[statevar+'_'+POLE]


# Drop unneeded points for speed 
if POLE == 'N':
  ds_s2s_state = ds_s2s_state.where(ds_s2s_state.lat>=60,drop=True)
else:
  ds_s2s_state = ds_s2s_state.where(ds_s2s_state.lat<=-60,drop=True)


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

# colormap and levels
levels = np.array([0,0.1,0.5,1,1.5,2,3,5,10])

# Extract a subset of the colormap
original_cmap = plt.cm.inferno
subset_colors = original_cmap(np.linspace(0.05, 1, 256))  # Adjust the range as needed
cmap = LinearSegmentedColormap.from_list('subset_cmap', subset_colors)

# Create figures
fig, axes = plt.subplots(2, 3, figsize=(16, 12), constrained_layout=True,
                         subplot_kw={'projection': proj},
                         gridspec_kw={'hspace': 0.01, 'wspace': 0.06})
fig.suptitle(f'{statevar_name} Scenarios',fontsize=15)

for i, ax in enumerate(fig.axes):
  if (POLE=='N'):
    ax.set_extent([-180, 180, 45, 90], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='darkgrey',zorder=0) 
  else:
    ax.set_extent([-180, 180, -45, -90], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='darkgrey', zorder=0)
    ax.add_feature(ice_shelves, facecolor='darkgrey',zorder=0)
  
  gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, y_inline=True,
          xlocs=range(-180,181,30), ylocs=(-60,60), ylim=(-60,60), rotate_labels=False,
          linewidth=1, color='dimgrey', alpha=0.5)
          
  gl.xlabel_style = {'size': 12}
  gl.ylabel_style = {'size': 12}
  gl.ylabel_style = {'rotation': -25}
  gl.left_labels=False

  # fix for cartopy draw error:
  plt.draw()
  for ea in gl.right_label_artists:
      ea.set_visible(True)
  for ea in gl.ylabel_artists:
    pos = ea.get_position()
    if POLE=='N':
      ea.set_position([150, pos[1]-2])
    else:
      ea.set_position([30, pos[1]+2])


for i, (sea,mon_name) in enumerate(zip(SEASON,MONTH)):
  
  mon = int(sea[-2:])
  # print(mon)
  
  ds_s2s_state_mon = ds_s2s_state.sel(time=ds_s2s_state.time.dt.month == mon)

  for j, scenario in enumerate(scenarios):
    # print(scenario)
    times_s2s_file = 'interm_data/times_data/'+'timess2s' + '_' + statevar + '_' + POLE  + '_'+ mon_name + '_' + scenario + '.npy'
    times = np.load(times_s2s_file)
    # print(len(times))
    ds_s2s_state_scen = ds_s2s_state_mon.sel(time=times) # Select times for each scenario
    ds_s2s_state_scen = ds_s2s_state_scen.mean(dim='time') # Take average across time
    # print(ds_s2s_state_scen)

    #Convert from kg/m2-s to mm/day
    ds_s2s_state_scen = ds_s2s_state_scen*86400
    # print(ds_s2s_state_scen)
    
    # ds_s2s_test = ds_s2s_state_scen.where(ds_s2s_state_scen!=0)
    # print(ds_s2s_test.mean(dim=['lat','lon']))
  
    plot = ds_s2s_state_scen.plot.contourf(ax=axes[i,j],transform=transform,vmin=0, vmax=12, levels=levels,cmap=cmap,extend='max',zorder=1,add_colorbar=False)
    axes[i,0].set_yticks([])
    axes[i,0].set_ylabel(mon_name,fontsize=20, rotation='horizontal',labelpad=40,weight='bold')
    axes[i,j].set_title('')
    axes[0,j].set_title(scenario,fontsize=22)

cbar = fig.colorbar(plot, ax=axes.ravel().tolist(),orientation='horizontal',shrink=0.8,ticks=levels,extend='max')
cbar.ax.tick_params(labelsize=18)
cbar.set_label('Snowfall (mm/day)',size=20)
fig.savefig(PLOT_PATH+pngname)