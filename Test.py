#! /usr/bin/env python
#Throwaway file for testing code

import numpy as np
import numpy.ma as ma
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
from netCDF4 import Dataset,num2date,date2num
import cmocean as cmocean
import cartopy
from cartopy import crs as ccrs, feature as cfeature  
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import xarray as xr
xr.set_options(display_width=95, display_max_rows=40, display_expand_attrs=False)
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import dask 
dask.config.set({"array.slicing.split_large_chunks": True}) 
import glob
import datetime as dt
import time
import sys
import os
# import re
# import struct

# Define misc variables
COLLECTION='geosgcm_surf'
COLLECTION2='geosgcm_seaice'
EXPDIR='/nobackup/hzafar/Ys2s3_base_9'
# EXPDIR=sys.argv[1] #extract experiment location from arguments
HOMDIR=EXPDIR
EXPID=EXPDIR.split('/')[-1]
PLOT_PATH = './plots/'
POLE = 'N'
# POLE=sys.argv[1] #extract pole of interest from arguments
pngname = EXPID+'_aice_seasonal_diff_'+POLE
cmap_ice = cmocean.cm.ice
levels_aice = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                     0.8, 0.9, 0.95, 0.99])
darkest_cmap = cmap_ice(0)
transform = crs=ccrs.PlateCarree()

SEASON=['M03']
MONTH=['MAR']
proj = ccrs.NorthPolarStereo()

fig, axes = plt.subplots(2,figsize=(6,10), constrained_layout=True, subplot_kw={'projection': proj})
for i, ax in enumerate(fig.axes):
  if (POLE=='N'):
    ax.set_extent([-180, 180, 45, 90], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='darkgrey', zorder=1) 
  else:
    ax.set_extent([-180, 180, -45, -90], crs=ccrs.PlateCarree())
    ax.add_feature(land,facecolor='darkgrey',zorder=1)
    ax.add_feature(ice_shelves, facecolor='darkgrey',zorder=1)


filename='HadISST.2.2.0.0_sea_ice_concentration.nc'
fname='./ObservationData/HADISST2/'+filename
ds_Had = xr.open_dataset(fname)
ds_Had = ds_Had.rename({'latitude': 'lat', 'longitude':'lon'}) #rename to match S2S
ds_Had.coords['lon'] = (ds_Had.coords['lon'] + 180) % 360 - 180 #convert to -180 to 180
ds_Had = ds_Had.sortby(ds_Had.lon)
ds_Had_selyr = ds_Had.sel(time=slice("1990-01-01", "2010-12-31")).groupby('time.month').mean()

for i, (sea,mon) in enumerate(zip(SEASON,MONTH)):
    #Extract HadISST month and plot
    mon_num = int(sea[-2:])
    ds_Had_selmon = ds_Had_selyr.sel(month=mon_num)
    ds_Had_selmon['sic'].plot.pcolormesh(ax=axes[0], transform=transform)

    # print(ds_Had_selmon)
    #Extract S2S data
    fname=EXPDIR+'/'+COLLECTION+'/'+EXPID+'.'+COLLECTION+'.monthly.clim.'+sea+'.nc4'
    # '/nobackup/hzafar/Ys2s3_base_9/geosgcm_surf/Ys2s3_base_9.geosgcm_surf.monthly.clim.M03.nc4'
    if os.path.isfile(fname): #if climatology file exists
        ds_s2s = xr.open_dataset(fname)

    else: #if no climatology file, compute one from the monthly files
        sys.exit("ERROR CLIM FILE") #for now, if no file, produce error

    variables = ['FRSEAICE','FROCEAN'] #Extract wanted variables
    ds_s2s = ds_s2s[variables].mean(dim='time')
    ds_s2s['FRSEAICE'].plot.pcolormesh(ax=axes[1], transform=transform)

fig.savefig(PLOT_PATH+'testplot.png')

###AICE ANOM
'''
# Define misc variables
COLLECTION ='geosgcm_surf'
COLLECTION2 = 'geosgcm_prog'
COLLECTION3 ='geosgcm_seaice'
EXPDIR='/nobackup/hzafar/Ys2s3_base_9'
HOMDIR=EXPDIR
EXPID=EXPDIR.split('/')[-1]
PLOT_PATH = './plots/'
POLE= 'N' #sys.argv[2] #extract pole of interest from arguments
# pngname = EXPID+'_AICE_ANOM_'+POLE
# scenarios = ['High SLP', 'Low SLP', 'Control SLP']

# if POLE == 'N': 
#   SEASON=['M03', 'M09']
#   MONTH=['MAR', 'SEP']
# else:
#   SEASON=['M09', 'M02']
#   MONTH=['SEP', 'FEB']
sea = 'M03'
mon = 'MAR'

# Import HadISST2
filename='HadISST.2.2.0.0_sea_ice_concentration.nc'
fileloc='./ObservationData/HADISST2/'+filename
ds_Had = xr.open_mfdataset(fileloc)
ds_Had = ds_Had.rename({'latitude': 'lat', 'longitude':'lon'}) #rename to match S2S
ds_Had.coords['lon'] = (ds_Had.coords['lon'] + 180) % 360 - 180 #convert to -180 to 180
ds_Had = ds_Had.sortby(ds_Had.lon)
ds_Had_selyr = ds_Had.sel(time=slice("1990-01-01", "2010-12-31")) #select 1990-2010

# Import MERRA2 analysis from discover (to classify HadISST)
filename = 'MERRA2_extract_SLP_'+POLE+'.nc'
fileloc = './transfer/' + filename
ds_MERRA = xr.open_dataset(fileloc)

mon_num = int(sea[-2:])
ds_Had_selmon = ds_Had_selyr.sel(time=ds_Had_selyr.time.dt.month == mon_num)
ds_MERRA_selmon = ds_MERRA.sel(time=ds_MERRA.time.dt.month == mon_num)

# Import S2S Seaice data for month
variables = ['FRSEAICE','FROCEAN'] #define wanted variables
ds_s2s = xr.open_mfdataset(
    EXPDIR+'/'+COLLECTION+'/'+EXPID+'.'+COLLECTION+'.monthly.????'+sea[-2:]
    +'.nc4',parallel=True)[variables]
# ds = xr.merge([xr.open_dataset(f) for f in glob.glob(EXPDIR+'/'+COLLECTION+'/'+EXPID+'.'+COLLECTION+'.monthly.????'+sea[-2:]+'.nc4')])

# Drop unneeded points for speed
if POLE == 'N':
    ds_s2s = ds_s2s.where(ds_s2s.lat>=30,drop=True)
    ds_Had_selmon = ds_Had_selmon.where(ds_Had.lat>=30,drop=True)
else:
    ds_s2s = ds_s2s.where(ds_s2s.lat<=-30,drop=True)
    ds_Had_selmon = ds_Had_selmon.where(ds_Had_selyr.lat<=-30,drop=True)

ds_s2s = ds_s2s.where(ds_s2s.lat>=30,drop=True)
ds_Had_selmon = ds_Had_selmon.where(ds_Had.lat>=30,drop=True)

## Extract state variable of interest/analysis for S2S
### Note MERRA2 analysis was done on discover
state_var = ['SLP']
ds_s2s_state = xr.open_mfdataset(
EXPDIR+'/'+COLLECTION2+'/'+EXPID+'.'+COLLECTION2+'.monthly.????'+sea[-2:]
+'.nc4',parallel=True)[state_var]
if POLE == 'N':
    ds_s2s_state = ds_s2s_state.where(ds_s2s.lat>=60,drop=True)
else:
    ds_s2s_state = ds_s2s_state.where(ds_s2s.lat<=-60,drop=True)

ds_s2s_state = ds_s2s_state.mean(dim=["lat","lon"])

# State variable analysis
## Compute means and std for state variables for MERRA2 and S2S
mean_MER = ds_MERRA_selmon.mean(dim='time')
std_MER = ds_MERRA_selmon.std(dim='time')
mean_s2s = ds_s2s_state.mean(dim='time')
std_s2s = ds_s2s_state.std(dim='time')

## Extract time arrays for high/low/control scenarios
times_MER_high = ds_MERRA_selmon.where(ds_MERRA_selmon > (mean_MER+std_MER), 
    drop=True).coords['time'].values
# times_MER_low = ds_MERRA_selmon.where(ds_MERRA_selmon < (mean_MER-std_MER), 
#       drop=True).coords['time'].values
# times_Had = ds_MERRA_selmon.where((ds_MERRA_selmon <= (mean_MER+std_MER))
#     & (ds_MERRA_selmon >= (mean_MER-std_MER)), drop=True).coords['time'].values

times_s2s_high = ds_s2s_state.where(ds_s2s_state > (mean_s2s+std_s2s), 
    drop=True).coords['time'].values
# times_s2s_low = ds_s2s_state.where(ds_s2s_state < (mean_s2s-std_s2s), 
#         drop=True).coords['time'].values
# times_s2s = ds_s2s_state.where((ds_s2s_state <= (mean_s2s+std_s2s))
#         & (ds_s2s_state >= (mean_s2s-std_s2s)),drop=True).coords['time'].values

## Select and compute S2S and HadISST data for high/low/control scenarios
ds_Had_scen = ds_Had_selmon.sel(time=times_MER_high, method='nearest').mean(dim='time')
print(ds_Had_scen)
ds_s2s_scen = ds_s2s.sel(time=times_s2s_high, method='nearest').mean(dim='time')
print(ds_s2s_scen)
ds_s2s_sceni = ds_s2s_scen.interp_like(ds_Had_scen)
xr.align(ds_s2s_sceni, ds_Had_scen, join='exact') #Check coordinates match (gives error if not)
ds_s2s_sceni = ds_s2s_sceni.assign(AICE=ds_s2s_sceni['FRSEAICE']/ds_s2s_sceni['FROCEAN'])
ds_s2s_sceni = ds_s2s_sceni.assign(DIFF_Had=ds_s2s_sceni['AICE'] - ds_Had_scen['sic'])
print(ds_s2s_sceni)
sys.exit()

'''
###AICE OR DIFF####
'''
COLLECTION='geosgcm_seaice'
COLLECTION2='geosgcm_surf'
EXPDIR='/nobackup/hzafar/Ys2s3_base_9' #sys.argv[1] #extract experiment location from arguments
HOMDIR=EXPDIR
EXPID=EXPDIR.split('/')[-1]
PLOT_PATH = './plots/'

POLE= 'N' #sys.argv[2] #extract pole of interest from arguments
pngname = EXPID+'_AICE_CLIM_'+POLE

# Define parameters for N and S poles and plotting
if POLE == 'N': #climatology for N Pole
  SEASON=['M03', 'M09']
  MONTH=['MAR', 'SEP']
  proj = ccrs.NorthPolarStereo()
  
else: #climatology for S pole
  SEASON=['M09', 'M02']
  MONTH=['SEP', 'FEB']
  proj = ccrs.SouthPolarStereo()

transform = crs=ccrs.PlateCarree()
# Generate figure
fig, axes = plt.subplots(2, figsize=(10, 16), subplot_kw={'projection': proj})


#Extract HadISST2
filename='HadISST.2.2.0.0_sea_ice_concentration.nc'
fname='./ObservationData/HADISST2/'+filename
ds_Had = xr.open_dataset(fname)
ds_Had = ds_Had.rename({'latitude': 'lat', 'longitude':'lon'}) #rename to match S2S
ds_Had.coords['lon'] = (ds_Had.coords['lon'] + 180) % 360 - 180 #convert to -180 to 180
ds_Had = ds_Had.sortby(ds_Had.lon)
ds_Had_selyr = ds_Had.sel(time=slice("1990-01-01", "2010-12-31")).groupby('time.month').mean()
'''
##### DIFF PLOTS ##########
'''
# Loop over figures
i = 0
for sea,mon in zip(SEASON,MONTH):
  #Extract HadISST month:
  mon_num = int(sea[-2:])
  ds_Had_selmon = ds_Had_selyr.sel(month=mon_num)
  # print(ds_Had_selmon)

  #Extract S2S data
  fname=EXPDIR+'/'+COLLECTION2+'/'+EXPID+'.'+COLLECTION2+'.monthly.clim.'+sea+'.nc4'
  # '/nobackup/hzafar/Ys2s3_base_9/geosgcm_surf/Ys2s3_base_9.geosgcm_surf.monthly.clim.M03.nc4'
  #print fname
  if os.path.isfile(fname): #if climatology file exists
    ds_s2s = xr.open_dataset(fname)

  else: #if no climatology file, compute one from the monthly files
    #for now, if no file, produce error
    sys.exit("ERRROR CLIM FILE")

  #  Extract wanted variables
  variables = ['FRSEAICE','FROCEAN']
  ds_s2s = ds_s2s[variables]
  # print(ds_s2s)
  
  ds_s2s_interp = ds_s2s.interp_like(ds_Had_selmon)
  # print(ds_s2s_interp)

  diff_ice = ds_s2s['FRSEAICE'] - ds_Had_selmon['sic']
  diff_ice.plot(ax=axes[i])

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
        # ea.set_visible(False)
        
  i += 1
plt.show()
'''

# Geosgcm_seaice tests:
'''
# Replace dummy lat/lon with LAT/LON
# fname=EXPDIR+'/'+COLLECTION+'/'+EXPID+'.'+COLLECTION+'.monthly.clim.M03.nc4'
et = time.time()
fnames=EXPDIR+'/'+COLLECTION+'/'+EXPID+'.'+COLLECTION+'.monthly.??????.nc4'
variables = ['LAT','LON','AICE', 'HICE','TMASK', 'AREA'] #define wanted variables
drop_vars = ['DAIDTD', 'DIVU', 'DRAFT', 'DRAFT0', 'DVIDTD', 'DVIRDGDT', 'FHOCN', 'FRESH', 'FROCEAN', 'FSALT', 'HICE0', 'HSNO', 'HSNO0', 'SHEAR', 'SLV', 'SSH', 'STRCORX', 'STRCORY', 'STRENGTH', 'STRINTX', 'STRINTY', 'STRTLTX', 'STRTLTY', 'TAUXBOT', 'TAUXI', 'TAUXIB', 'TAUXOCNB', 'TAUYBOT', 'TAUYI', 'TAUYIB', 'TAUYOCNB','UI', 'UOCN', 'VEL', 'VI', 'VOCN', 'Var_AICE', 'Var_AREA', 'Var_DAIDTD', 'Var_DIVU', 'Var_DRAFT', 'Var_DRAFT0', 'Var_DVIDTD', 'Var_DVIRDGDT', 'Var_FHOCN', 'Var_FRESH', 'Var_FROCEAN', 'Var_FSALT', 'Var_HICE', 'Var_HICE0', 'Var_HSNO', 'Var_HSNO0', 'Var_SHEAR', 'Var_SLV', 'Var_SSH', 'Var_STRCORX', 'Var_STRCORY', 'Var_STRENGTH', 'Var_STRINTX', 'Var_STRINTY', 'Var_STRTLTX', 'Var_STRTLTY', 'Var_TAUXBOT', 'Var_TAUXI', 'Var_TAUXIB', 'Var_TAUXOCNB', 'Var_TAUYBOT', 'Var_TAUYI', 'Var_TAUYIB', 'Var_TAUYOCNB', 'Var_TMASK', 'Var_UI', 'Var_UOCN', 'Var_VEL', 'Var_VI', 'Var_VOCN']
# '/nobackup/hzafar/Ys2s3_base_9/geosgcm_seaice/Ys2s3_base_9.geosgcm_seaice.monthly.clim.M03.nc4'
# ds_s2s = xr.open_mfdataset(fnames,concat_dim="time",combine="nested", drop_variables=drop_vars)
ds_s2s = xr.open_mfdataset(fnames, parallel = True,  chunks={'lat': 40, 'lon': 40},
    decode_times=True, concat_dim="time", combine="nested",drop_variables=drop_vars)

# 
# ds_s2s= xr.open_dataset(fname)
ds_s2s = ds_s2s.chunk({'time':200})
print(ds_s2s)
print(time.time() - et)
et = time.time()
#Replace dummy lat/lon with LAT/LON
LAT = ds_s2s['LAT'][0,:,0].values
LON = ds_s2s['LON'][0,0,:].values
# print(LAT.shape,LON.shape,sep='\n')
ds_s2s.coords['lat'] = LAT
ds_s2s.coords['lon'] = LON
print(time.time() - et)
et = time.time()
variables = ['HICE','AICE','TMASK','AREA']
ds_s2s = ds_s2s[variables]
# ds_s2s.load()
ds_s2s = ds_s2s.where((ds_s2s.TMASK>0.5) & (ds_s2s.HICE>0.06) & 
    (ds_s2s.AICE>0.15) & (ds_s2s.AICE<1.e10),drop=True)
ds_s2s = ds_s2s.fillna(0)
ds_bool = ds_s2s['AICE'].where(ds_s2s.AICE==0,other=1).rename('AICE_bool')
ds_area = ds_s2s['AREA']
ds_extent = xr.merge([ds_bool, ds_area])
ds_extent = ds_extent.assign(EXTENT=ds_extent.AICE_bool*ds_extent.AREA/(1000)**2) #m2 to km2
ds_extent = ds_extent.sum(dim=['lat','lon'])
print(ds_extent)
print(time.time() - et)

## Write data to netcdf for faster load later
ds_s2s.to_netcdf('interm_data/preprocessed_seaice_extent.nc')
sys.exit()
'''

# Geosgcm_surf tests:`
'''
fname='/nobackup/hzafar/Ys2s3_base_9/geosgcm_surf/Ys2s3_base_9.geosgcm_surf.monthly.clim.M03.nc4'
ncfile = Dataset(fname, 'r', format='NETCDF4')
lat=ncfile.variables['lat'][:] #Latitudes
lon=ncfile.variables['lon'][:] #Longitudes
fraci=ncfile.variables['FRSEAICE'][:] 
fraco=ncfile.variables['FROCEAN'][:]
# hice=ncfile.variables['HICE'][:] #Mean ice thickness of grid cell
# aice=ncfile.variables['AICE'][:] #Ie concentration of grid cell
# tmask=ncfile.variables['TMASK'][:] #Ocean mask (0=land, 1=ocean)
ncfile.close()
# print(lon)
print(lat)
print(lon)
'''

#####ANNCYCLE#####
'''
COLLECTION='geosgcm_seaice'
EXPDIR= '/nobackup/hzafar/Ys2s3_base_9' #sys.argv[1] #extract experiment location from arguments
HOMDIR=EXPDIR
EXPID=EXPDIR.split('/')[-1]
PLOT_PATH = './plots/'
pngname = EXPID+'EXTENT_ANNCYCLE'

years =['1990-2010','1995-2005','1998-2002']
start_year = np.array([1990,1995,1998],dtype=int)
end_year = np.array([2010,2005,2002],dtype=int)
fig, ax = plt.subplots(2, figsize=(10, 16))
linespec=['o-','o--']

filename='HadISST.2.2.0.0_sea_ice_concentration.nc'
fname='./ObservationData/HADISST2/'+filename
ds_Had = xr.open_dataset(fname, decode_times=True)
# print(ds_Had)

ds_Had = ds_Had.sic
# print(ds_Had)
# for startyr, endyr in zip(start_year,end_year):
startyr = 1990
endyr = 2010
start_date=str(startyr)+"-01-01"
end_date=str(endyr)+"-12-31"
Had_sel_avg = ds_Had.sel(time=slice(start_date, end_date)).groupby('time.month').mean()
# print(Had_sel_avg)
Had_mask = Had_sel_avg.where(Had_sel_avg > 0.15)
# print(Had_mask)

POLE = 'S'
if POLE=='N':
    Had_mask = Had_mask.where(Had_sel_avg.latitude > 0)
    Had_mask = Had_mask.fillna(0)
    Had_bool = Had_mask.where(Had_mask == 0, other=1)
    # print(Had_bool)
if POLE=='S':
    Had_mask = Had_mask.where(Had_sel_avg.latitude < 0)
    Had_mask = Had_mask.fillna(0)
    Had_bool = Had_mask.where(Had_mask == 0, other=1)

#Calculate extent
R_earth = 6357 #6378 #km
Had_ext = 1.e-6*Had_bool*np.pi/180*(R_earth)**2*np.abs(
    np.sin(np.deg2rad(Had_bool.latitude+0.5))-np.sin(np.deg2rad(Had_bool.latitude-0.5)))
Had_ext = Had_ext.sum(dim=['latitude', 'longitude']) 
print(Had_ext)

# area_grid = np.pi/180*(R_earth)**2*np.abs(
#     np.sin(Had_ext.latitude.+0.5)-np.sin(Had_ext.latitude)-0.5))
# Had_ext.assign(area_grid = lambda Had_ext.latitude: np.pi/180*(R_earth)**2*np.abs(
# np.sin(Had_ext.latitude.+0.5)-np.sin(Had_ext.latitude)-0.5))
# lat = ds_Had.latitude.values
# R_earth = 6378 #km
# area_grid=np.pi/180*(R_earth)**2*np.abs(np.sin(lat+0.5)-np.sin(lat-0.5))
# area_grid = np.array([area_grid]*3).T
# area_grid = np.repeat(area_grid.reshape(180,360),360,axis=1)


for k,POLE in enumerate(['N', 'S']):
#Extract HADISST2 extent
    fname=glob.glob('./ObservationData/HADISST2/HadISST.2.2.1.0_'+POLE+'*_'+'sea_ice_extent.txt')
    # print(fname)
    if POLE == 'S':
        ds_Had = pd.read_csv(fname[0],delim_whitespace=True, skiprows=1)
    else: 
        ds_Had = pd.read_csv(fname[0],delim_whitespace=True)
    ds_Had.columns = ['Year', 'Month', 'Extent']
    # ds_Had['datetime']=pd.to_datetime(ds_Had.assign(Day=1)[['Year','Month','Day']])
    # print(ds_Had)
    i=0
    for startyr, endyr in zip(start_year,end_year):
        start_date=str(startyr)+"-01-01"
        end_date=str(endyr)+"-12-31"
        ds_sel_yr = ds_Had[ds_Had['Year'].between(startyr, endyr)]
        # print(ds_selyr)
        ds_mon_avg = ds_sel_yr.groupby('Month')['Extent'].mean()
        # ds_mon_avg.index = pd.to_datetime(ds_mon_avg.index, format='%m')
        print(ds_mon_avg.values)
    
        ds_mon_avg.plot(ax=ax[k], label=years[i])   # 'b'+linespec[k]
        # print(ds_mon_avg)
        # ds_mon_avg = ds_mon_avg*1.e-6
        # plt.show()
        i += 1
        
    # ax[k].set_xlim(1,12)
    # ax[k].xaxis.set_major_locator(mdates.MonthLocator(range(1,13), bymonthday=15, interval=1))
    ax[k].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    # plt.xticks(rotation='vertical')
    # ax[k].set_xlabel('Month')
    # ax[k].set_ylabel('Extent ($10^{6}$ $km^{2}$)')
    ax[k].legend()
    ax[k].grid(True)
    plt.show()

# plt.show()

for k,POLE in enumerate(['N', 'S']):
    for mon in range(1,13,1):
        SEASON='M'+str(mon)
        if mon < 10:
            SEASON='M0'+str(mon)
        filename=EXPID+'.'+COLLECTION+'.monthly.clim.'+SEASON+'.nc4'
        fname=EXPDIR+'/'+COLLECTION+'/'+filename
        # print(fname)
        # total_extent[mon-1] = compute_extent(fname, area, POLE)
        # def compute_extent(fname,area,POLE): 
        ncfile = Dataset(fname, 'r', format='NETCDF4')
        lon=ncfile.variables['LON'][:]
        lat=ncfile.variables['LAT'][:]
        aice=ncfile.variables['AICE'][:]  #Ice concentration of grid cell
        hice=ncfile.variables['HICE'][:] #Mean ice thickness of grid cell
        fro=ncfile.variables['FROCEAN'][:]
        tmask=ncfile.variables['TMASK'][:] #Ocean mask (0=land, 1=ocean)
        area=ncfile.variables['AREA'][:]  #Area of grid cell (m2)
        ncfile.close()
        # atemp=area*fro[0]

        aicem=ma.masked_where(tmask[0,:,:]<0.5, aice[0,:,:])
        aicem=ma.masked_where(aicem<0.15, aicem)
        # aicem=ma.masked_where(aicem>1.e10, aicem)
        # aicem=ma.masked_where(hice[0,:,:]<0.06,  aicem)

        area_km = area[0,:,:]/(1000)**2 #m2 to km2
        if POLE=='N':
            aicem=ma.masked_where(lat<0.0, aicem)
        if POLE=='S':
            aicem=ma.masked_where(lat>0.0, aicem)
        extent[mon-1]=np.sum(aicem*area_km)
        # print(extent)
    extent = extent*1.e-6
    print(extent)

    months = np.arange(1,13,1)
    x_axis = np.zeros(12)
    for j in months:
	    x_axis[j-1]=mdates.date2num(datetime.date(1900,j,1)) 
    print(x_axis)
    ax[k].plot(x_axis,extent,c='k', label='Base9',lw=3)

    ax[k].xaxis.set_major_locator(mdates.MonthLocator())
    ax[k].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    # ax[k].set_xlabel('Month')
    # ax[k].set_ylabel('Extent ($10^{6}$ $km^{2}$)')
    plt.show()
'''