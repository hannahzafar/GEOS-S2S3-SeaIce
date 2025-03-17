#! /usr/bin/env python
# This code generates seasonal ice fraction spatial plots for the selected hemisphere
# Running this code requires the following arguments:  ['N' or 'S']
# BASE9: /nobackup/hzafar/Ys2s3_base_9

from netCDF4 import Dataset,num2date,date2num
import numpy as np
import numpy.ma as ma
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import xarray as xr
import cartopy
from cartopy import crs as ccrs, feature as cfeature  
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import warnings
warnings.filterwarnings('ignore')
import cmocean as cmocean
import glob
import struct
import datetime
import time
import sys
import os
import re

# Define input/output locations
COLLECTION='geosgcm_seaice'
EXPDIR='/nobackup/hzafar/Ys2s3_base_9' 
# EXPDIR=sys.argv[1] #extract experiment location from arguments
HOMDIR=EXPDIR
EXPID=EXPDIR.split('/')[-1]
PLOT_PATH = './plots/'
POLE=sys.argv[1] #extract pole of interest from arguments
# POLE=sys.argv[2] #extract pole of interest from arguments
pngname = EXPID+'_aice_seasonal_'+POLE

# Define climatology and plotting settings for N and S poles
if POLE == 'N': 
  SEASON=['M03', 'M09']
  MONTH=['MAR', 'SEP']
  proj = ccrs.NorthPolarStereo()

else:
  SEASON=['M09', 'M02']
  MONTH=['SEP', 'FEB']
  proj = ccrs.SouthPolarStereo()

transform = crs=ccrs.PlateCarree()

# Import Earth Features
ice_shelves = cfeature.NaturalEarthFeature(
        category='physical',
        name='antarctic_ice_shelves_polys',
        scale='10m')

land = cfeature.NaturalEarthFeature(
        category='physical',
        name='land',
        scale='10m')

#Truncate colormap
cmap = cmocean.cm.ice
aice_levels = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                     0.8, 0.9, 0.95, 0.99])
darkest_cmap = cmap(0)

# OLD
  # Extract FROCEAN for land surfaces
  # COLLECTION_surf = 'geosgcm_surf'
  # fname_surf=EXPDIR+'/'+COLLECTION_surf +'/'+EXPID+'.'+COLLECTION_surf +'.monthly.clim.'+'M09'+'.nc4'
  # ncfile_surf= Dataset(fname_surf, 'r', format='NETCDF4')
  # lon_surf=ncfile_surf.variables['lon'][:]
  # lat_surf=ncfile_surf.variables['lat'][:]
  # # landice=ncfile_surf.variables['FRLANDICE'][0,:,:]
  # frocean = ncfile_surf.variables['FROCEAN'][0,:,:]
  # ncfile_surf.close()
  # lon_surf2d, lat_surf2d = np.meshgrid(lon_surf, lat_surf)
  # # print(lat_surf2d.shape,lon_surf2d.shape, frocean.shape)
  # lon_surf2d=ma.masked_where(frocean>0.5,lon_surf2d)
  # lat_surf2d=ma.masked_where(frocean>0.5,lat_surf2d)
  # frocean=ma.masked_where(frocean>0.5,frocean)

#Extract HadISST2
filename='HadISST.2.2.0.0_sea_ice_concentration.nc'
fname='./ObservationData/HADISST2/'+filename
ds_Had = xr.open_dataset(fname)
ds_Had = ds_Had.rename({'latitude': 'lat', 'longitude':'lon'}) #rename to match S2S
ds_Had.coords['lon'] = (ds_Had.coords['lon'] + 180) % 360 - 180 #convert to -180 to 180
ds_Had = ds_Had.sortby(ds_Had.lon)
ds_Had_selyr = ds_Had.sel(time=slice("1990-01-01", "2010-12-31")).groupby('time.month').mean()

# Generate figure/axes and set plotting parameters
fig, axes = plt.subplots(2, figsize=(10, 16), subplot_kw={'projection': proj})
for i, ax in enumerate(fig.axes):
  if (POLE=='N'):
    ax.set_extent([-180, 180, 45, 90], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='darkgray',zorder=1) 
    ax.add_feature(cfeature.OCEAN, facecolor=darkest_cmap, zorder=0) 

  else:
    ax.set_extent([-180, 180, -45, -90], crs=ccrs.PlateCarree())
    # ax.add_feature(cfeature.LAND, facecolor='lightgrey', zorder=1)
    ax.add_feature(land,facecolor='darkgray')
    ax.add_feature(ice_shelves, facecolor='darkgray')
    # ax.add_feature(cfeature.COASTLINE,lw=1)
    ax.add_feature(cfeature.OCEAN, facecolor=darkest_cmap, zorder=0) 

  gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, y_inline=False,
          xlocs=range(-180,181,30), ylocs=range(-180,181,30), rotate_labels=False,
          linewidth=0.5, color='white', alpha=0.5)
  gl.xlabel_style = {'size': 15}
  gl.ylabel_style = {'size': 15}
  gl.right_labels=False
  # fix for cartopy draw error:
  plt.draw()
  for ea in gl.left_label_artists:
      ea.set_visible(True)


# Loop over figures
i = 0
for sea,mon in zip(SEASON,MONTH):

  # Select month of interest
  mon_num = int(sea[-2:])
  
  # Extract HADISST2 for month
  ds_Had_selmon = ds_Had_selyr.sel(month=mon_num)
  sic = ds_Had_selmon["sic"]
  # print(sic)
  # sys.exit()

  '''
  ## CAN'T GET XARRAY TO WORK???
  # Extract S2S Climatology Data for month
  variables = ['LAT', 'LON','AICE','TMASK'] #define wanted variables
  fname=EXPDIR+'/'+COLLECTION+'/'+EXPID+'.'+COLLECTION+'.monthly.clim.'+sea+'.nc4'
  ds_s2s = xr.open_mfdataset(fname)[variables]
  ds_s2s.load()
  print(ds_s2s)
  sys.exit()
  #Replace tripolar lat/lon with LAT/LON
  LAT = ds_s2s['LAT'][:,0].values
  LON = ds_s2s['LON'][0,:].values
  ds_s2s.coords['lat'] = LAT
  ds_s2s.coords['lon'] = LON

  ds_s2s = ds_s2s.where(ds_s2s.TMASK>0.5) #Mask over land
  ds_s2s = ds_s2s.mean(dim='time') #avg over one time for plotting
  # print(ds_s2s)
  # sys.exit()

  # Drop unneeded points for speed
  if POLE == 'N':
    ds_s2s = ds_s2s.where(ds_s2s.lat>=30,drop=True)
    sic = sic.where(sic.lat>=30,drop=True)
  else:
    ds_s2s = ds_s2s.where(ds_s2s.lat<=-30,drop=True)
    sic = sic.where(sic.lat<=-30,drop=True)
  print(ds_s2s)
  sys.exit()
  '''
  
  # OLD extract data:
  fname=EXPDIR+'/'+COLLECTION+'/'+EXPID+'.'+COLLECTION+'.monthly.clim.'+sea+'.nc4'
  ncfile = Dataset(fname, 'r', format='NETCDF4')
  if os.path.isfile(fname): #if climatology file exists
    fbot=ncfile.variables['HICE'][:] #Mean ice thickness of grid cell
    lon=ncfile.variables['LON'][:] #Longitudes
    lat=ncfile.variables['LAT'][:] #Latitudes
    aice=ncfile.variables['AICE'][:] #Ice concentration of grid cell
    tmask=ncfile.variables['TMASK'][:] #Ocean mask (0=land, 1=ocean)
    ncfile.close()
  
  else: #if no climatology file, compute one from the monthly files
     files = glob.glob(EXPDIR+'/'+COLLECTION+'/*monthly.????'+sea[-2:]+'.nc4')
     files.sort()
     ncfile = Dataset(files[0], 'r', format='NETCDF4')
     lon=ncfile.variables['LON'][:]
     lat=ncfile.variables['LAT'][:]
     tmask=ncfile.variables['TMASK'][:]
     ncfile.close()
     aice=np.zeros((1, tmask.shape[1], tmask.shape[2]))
     fbot=np.zeros((1, tmask.shape[1], tmask.shape[2]))
     for f in files:
       ncfile = Dataset(f, 'r', format='NETCDF4')
       hi=ncfile.variables['HICE'][:]
       ai=ncfile.variables['AICE'][:]
       ncfile.close()
       fbot += hi
       aice += ai
     fbot /= float(len(files))
     aice /= float(len(files))

  # Mask over land
  aicem = ma.masked_where(tmask<0.5, aice)
  fbotm = ma.masked_where(tmask<0.5, fbot)

  # OLD plot
    # ## Set map extents and grids/labels
    # if (POLE=='N'):
    #   axes[i].set_extent([-180, 180, 45.5, 90], crs=ccrs.PlateCarree())
    #   gl = axes[i].gridlines(crs=ccrs.PlateCarree(), draw_labels=True, y_inline=False,
    #                 xlocs=range(-180,181,30), ylocs=range(-180,181,30), rotate_labels=False,
    #                 linewidth=0.5, color='white', alpha=0.5)
    #   gl.right_labels=False
    #   gl.left_labels=True
    #   plt.draw()
    #   for ea in gl.left_label_artists:
    #       ea.set_visible(True)
    #   for ea in gl.right_label_artists:
    #       ea.set_visible(False)
        
    #   axes[i].add_feature(cfeature.LAND, facecolor='lightgrey',edgecolor='k')

    # if  (POLE=='S'):
    #   axes[i].set_extent([-180, 180, -45.5, -90], crs=ccrs.PlateCarree())
    #   gl = axes[i].gridlines(crs=ccrs.PlateCarree(), draw_labels=True, y_inline=False,
    #                 xlocs=range(-180,181,30), ylocs=range(-180,181,30), rotate_labels=False,
    #                 linewidth=0.5, color='white', alpha=0.5)
    #   gl.right_labels=False
    #   gl.left_labels=True
    #   plt.draw()
    #   for ea in gl.left_label_artists:
    #       ea.set_visible(True)
    #   for ea in gl.right_label_artists:
    #       ea.set_visible(False)

    #   axes[i].add_feature(cfeature.LAND, facecolor='lightgrey')
    #   axes[i].add_feature(land,facecolor='lightgrey')
    #   axes[i].add_feature(ice_shelves, facecolor='lightgrey')
    #   axes[i].add_feature(cfeature.COASTLINE)
  
  # Plot S2S ice concentration (AICE): contours and extent
  h_level = np.array([0.15]) #extent: at least 15% ice fraction
  ct_aice =  axes[i].contour(lon,lat,aicem[0,:,:], h_level, transform=transform, colors='lime', linewidths=3,zorder=2)
  aice = axes[i].contourf(lon,lat,aicem[0,:,:], aice_levels, 
                        transform=transform,cmap=cmap,vmin=0, vmax=0.99, extend='max')

  # Plot HadISST extent
  ct_sic = sic.plot.contour(ax=axes[i], levels=h_level, transform=transform, 
                          colors='m',linewidths=3,zorder=1)

  '''
  # Plot S2S from xarray
  ## Plot extent boundary
  ct_aice =  ds_s2s['AICE'].plot.contour(ax=axes[i], levels=h_level, transform=transform, 
                        colors='lime', linewidths=1.5)
  ## Plot contours
  aice = ds_s2s['AICE'].plot.contourf(ax=axes[i], levels=aice_levels,transform=transform,
                          cmap=cmap,vmin=0, vmax=0.99, extend='max',add_colorbar=False)
  '''

  # Add titles and legend 
  title=mon 
  axes[i].set_title(title,fontsize=28)
  h1,_ = ct_aice.legend_elements()
  h2,_ = ct_sic.legend_elements()
  # axes[i].legend([h1[0], h2[0]], ['HADISST2 Extent', 'GEOS-S2S-3 Extent'],loc='lower left',fontsize='18')

  i += 1

# Add colorbar
cbar = plt.colorbar(aice, ax=axes.ravel().tolist())
cbar.set_ticks(list(aice_levels))
cbar.ax.tick_params(labelsize=22)
cbar.set_label('Sea ice concentration',rotation=270, size=20, labelpad=24)
fig.legend([h1[0], h2[0]], ['GEOS-S2S-3 Extent','HADISST2 Extent'],loc='lower center',bbox_to_anchor=(0.5, 0),fontsize=20)

fig.savefig(PLOT_PATH+'/'+pngname, dpi=fig.dpi, bbox_inches='tight')
# plt.show()
