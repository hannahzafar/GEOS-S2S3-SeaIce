#! /usr/bin/env python
# This code generates sic plots for the selected hemisphere
# Running this code requires the following arguments: ['N' or 'S']
# BASE9 location: /nobackup/hzafar/Ys2s3_base_9

from netCDF4 import Dataset,num2date,date2num
import numpy as np
import numpy.ma as ma
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import xarray as xr
xr.set_options(display_width=95, display_max_rows=40, display_expand_attrs=False)
import cartopy
import warnings
warnings.filterwarnings('ignore')
from cartopy import crs as ccrs, feature as cfeature
import cmocean as cmocean
import glob
import struct
import datetime
import time
import sys
import os
import re

# Define misc variables
COLLECTION='geosgcm_surf'
COLLECTION2='geosgcm_seaice'
EXPDIR='/nobackup/hzafar/Ys2s3_base_9'
# EXPDIR=sys.argv[1] #extract experiment location from arguments
HOMDIR=EXPDIR
EXPID=EXPDIR.split('/')[-1]
PLOT_PATH = './plots/'
POLE=sys.argv[1] #extract pole of interest from arguments
pngname = EXPID+'_aice_seasonal_diff_'+POLE
cmap_ice = cmocean.cm.ice
levels_aice = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                     0.8, 0.9, 0.95, 0.99])
darkest_cmap = cmap_ice(0)
# labels = ['GEOS-S2S-3', 'HadISST2\n1990-2010', 'Difference']
labels = ['GEOS-S2S-3', 'HadISST2', 'Difference']

cmap_diff = plt.cm.RdBu_r
levels_diff = np.array([-0.8, -0.6, -0.4, -0.2, -0.1, 0.1, 0.2, 
                        0.4, 0.6, 0.8])

transform = crs=ccrs.PlateCarree()
if POLE == 'N': #climatology for N Pole
  SEASON=['M03', 'M09']
  MONTH=['MAR', 'SEP']
  proj = ccrs.NorthPolarStereo()
  
else: #climatology for S pole
  SEASON=['M09', 'M02']
  MONTH=['SEP', 'FEB']
  proj = ccrs.SouthPolarStereo()
  # Import ice shelves
  ice_shelves = cfeature.NaturalEarthFeature(
        category='physical',
        name='antarctic_ice_shelves_polys',
        scale='10m')

  land = cfeature.NaturalEarthFeature(
        category='physical',
        name='land',
        scale='10m')

# Generate figure/axes and set parameters
fig, axes = plt.subplots(2, 3, figsize=(18, 12), constrained_layout=True,
                         subplot_kw={'projection': proj},
                         gridspec_kw={'hspace': 0.1, 'wspace': 0.05})

for i, ax in enumerate(fig.axes):
  if (POLE=='N'):
    ax.set_extent([-180, 180, 45, 90], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='darkgrey', zorder=1) 
    

  else:
    ax.set_extent([-180, 180, -45, -90], crs=ccrs.PlateCarree())
    ax.add_feature(land,facecolor='darkgrey',zorder=1)
    ax.add_feature(ice_shelves, facecolor='darkgrey',zorder=1)

  # Gridlines
  # Last column gridlines
  if (i == 2) or (i == 5):
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, y_inline=False,
          xlocs=range(-180,181,30),  ylocs=(-60,60), ylim=(-60,60),  rotate_labels=False,
          linewidth=1, color='dimgrey', alpha=0.5)
  # Other gridlines
  else: 
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, y_inline=False,
            xlocs=range(-180,181,30),  ylocs=(-60,60), ylim=(-60,60), rotate_labels=False,
            linewidth=1, color='white', alpha=0.5)

  gl.xlabel_style = {'size': 12}
  gl.ylabel_style = {'size': 12}
  gl.left_labels=False


  # fix for cartopy draw error:
  plt.draw()
  for ea in gl.right_label_artists:
      ea.set_visible(True)

  # Add ocean for first two columns
  if (i != 2) & (i != 5):
    ax.add_feature(cfeature.OCEAN, facecolor=darkest_cmap, zorder=0)

# Extract HadISST2
filename='HadISST.2.2.0.0_sea_ice_concentration.nc'
fname='./ObservationData/HADISST2/'+filename
ds_Had = xr.open_dataset(fname)
ds_Had = ds_Had.rename({'latitude': 'lat', 'longitude':'lon'}) #rename to match S2S
ds_Had.coords['lon'] = (ds_Had.coords['lon'] + 180) % 360 - 180 #convert to -180 to 180
ds_Had = ds_Had.sortby(ds_Had.lon)
ds_Had_selyr = ds_Had.sel(time=slice("1990-01-01", "2010-12-31")).groupby('time.month').mean()
# print(ds_Had_selyr)
# ds_Had_selyr = ds_Had_selyr.where(ds_Had_selyr.sic>0.10)
# print(ds_Had_selyr)

# Plot values
for i, (sea,mon) in enumerate(zip(SEASON,MONTH)):
  #Extract HadISST month and plot
  mon_num = int(sea[-2:])
  ds_Had_selmon = ds_Had_selyr.sel(month=mon_num)
  # print(ds_Had_selmon)
  
  #Extract S2S data
  fname=EXPDIR+'/'+COLLECTION+'/'+EXPID+'.'+COLLECTION+'.monthly.clim.'+sea+'.nc4'

  if os.path.isfile(fname): #if climatology file exists
    ds_s2s = xr.open_dataset(fname)

  else: #if no climatology file, compute one from the monthly files
    sys.exit("ERROR CLIM FILE") #for now, if no file, produce error
  
  variables = ['FRSEAICE','FROCEAN'] #Extract wanted variables
  ds_s2s = ds_s2s[variables].mean(dim='time')

  ds_s2s_interp = ds_s2s.interp_like(ds_Had_selmon) #Interpolate lat/lon with HadISST
  #xr.align(ds_s2s_interp, ds_Had_selmon, join='exact') #Check coordinates match

  # Calculate AICE and difference
  ds_s2s_interp = ds_s2s_interp.assign(AICE=ds_s2s['FRSEAICE']/ds_s2s['FROCEAN'])
  ds_s2s_interp = ds_s2s_interp.assign(DIFF=ds_s2s_interp['AICE'] - ds_Had_selmon['sic']) #Generate difference plot
  # print(ds_s2s_interp)
  
  xr_plots = xr.merge([ds_s2s_interp[['AICE','DIFF']],ds_Had_selmon['sic']], join='exact')
  if POLE == 'N':
    xr_plots = xr_plots.sel(lat=slice(90,30))
  else:
    xr_plots = xr_plots.sel(lat=slice(-30,-90))

  #Plot data
  # plot = xr_plots['AICE'].where(xr_plots['AICE']>0.01).plot.contourf(ax=axes[i,0], levels=levels_aice, transform=transform, cmap=cmap_ice,add_colorbar=False)
  plot = xr_plots['AICE'].plot.pcolormesh(ax=axes[i,0], levels=levels_aice, transform=transform, cmap=cmap_ice,add_colorbar=False,zorder=1)
  xr_plots['sic'].plot.pcolormesh(ax=axes[i,1], levels=levels_aice, transform=transform, cmap=cmap_ice,add_colorbar=False,zorder=1)
  plot2 = xr_plots['DIFF'].plot.pcolormesh(ax=axes[i,2],levels=levels_diff, transform=transform, cmap=cmap_diff,extend='both',add_colorbar=False,zorder=1) #zorder=0 


  # Add month labels
  axes[i,0].set_yticks([])
  axes[i,0].set_ylabel(mon,fontsize=20, rotation='horizontal',labelpad=40, weight='bold')
    # for j in range(3):
    #   axes[i,j].set_title(mon,fontsize=18)
  for j in range(3):
      axes[i,j].set_title('')
      
#Title and colorbars
for j in range(3):
    title=f"{labels[j]}"
    axes[0,j].set_title(title,fontsize=22)

cbar = fig.colorbar(plot, ax=axes[1,:2], orientation="horizontal",ticks=levels_aice)
cbar.set_label('Sea ice concentration', size=20)
cbar.ax.tick_params(labelsize=18)
# ticks_cbar2=np.insert(levels_diff,5,0)
cbar2 = fig.colorbar(plot2, ax=axes[:,-1], orientation="vertical",shrink=1.05, ticks=levels_diff)
cbar2.ax.tick_params(labelsize=16)
minorticks = [0]
cbar2.set_ticks(minorticks, minor=True)
fig.savefig(PLOT_PATH+pngname, dpi=fig.dpi, bbox_inches='tight')
# plt.show()
