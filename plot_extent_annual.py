#!/usr/bin/env python
# This code generates Ice Extent time series plots for the annual cycle of sea ice
# in the NH and SH (both are shown on the same axis), compared to HadISST2.txt data.
# Running this code requires no additional arguments

from netCDF4 import Dataset,num2date,date2num
import numpy as np
import numpy.ma as ma
import matplotlib as mpl
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import cmocean as cmocean
import glob
import struct
import datetime
import time
import sys
# import os
# import re

# Define input/output locations
COLLECTION='geosgcm_seaice'
EXPDIR='/nobackup/hzafar/Ys2s3_base_9' 
# EXPDIR= #sys.argv[1] #extract experiment location from arguments
HOMDIR=EXPDIR
EXPID=EXPDIR.split('/')[-1]
PLOT_PATH = './plots/'
pngname = EXPID+'_ext_anncycle'
HAD_fileloc = './ObservationData/HADISST2/'
S2S_fileloc = './interm_data/'


# Define years for HADISST2
startyr = 1990
endyr = 2010
start_date=str(startyr)+"-01-01"
end_date=str(endyr)+"-12-31"

# Create figure/axes
fig, ax = plt.subplots(1,2, figsize=(14,6))

months = np.arange(1,13)
monthlabel = [datetime.date(1900, mon, 1).strftime('%b') for mon in months]

# Import HADISST2.nc data
fname= HAD_fileloc+'HadISST.2.2.0.0_sea_ice_concentration.nc'
ds_Had = xr.open_dataset(fname, decode_times=True)
ds_Had = ds_Had.sic
Had_sel_avg = ds_Had.sel(time=slice(start_date, end_date)).groupby('time.month').mean()

# # Import S2S Preprocessed sea ice extent
# fname = S2S_fileloc + 'preprocessed_seaice_extent.csv'
# ds_s2s = pd.read_csv(fname,index_col=0, parse_dates=True)
# ds_s2s_avg = ds_s2s.groupby(ds_s2s.index.month).mean()*1e-6

for k,POLE in enumerate(['N', 'S']):

    # Extract/modify/plot HADISST extent txt files
    fname= HAD_fileloc + 'HadISST.2.2.1.0_'+POLE+'H_sea_ice_extent.txt'
    if POLE == 'S':
        ds_Had = pd.read_csv(fname,delim_whitespace=True, skiprows=1)
    else: 
        ds_Had = pd.read_csv(fname,delim_whitespace=True)
    ds_Had.columns = ['Year', 'Month', 'Extent']
    # ds_Had['datetime']=pd.to_datetime(ds_Had.assign(Day=1)[['Year','Month','Day']])
    ds_sel_yr = ds_Had[ds_Had['Year'].between(startyr, endyr)]
    ds_mon_avg = ds_sel_yr.groupby('Month')['Extent'].mean()
    # ds_mon_avg.index = pd.to_datetime(ds_mon_avg.index, format='%m')
    ax[k].plot(months,ds_mon_avg.values,label='HadISST2 (1990-2010 average)', c='m',lw=2,ls='dashed')

    extent=np.zeros(12)
    for mon in range(1,13,1):
        SEASON='M'+str(mon)
        if mon < 10:
            SEASON='M0'+str(mon)
        filename=EXPID+'.'+COLLECTION+'.monthly.clim.'+SEASON+'.nc4'
        fname=EXPDIR+'/'+COLLECTION+'/'+filename
        ncfile = Dataset(fname, 'r', format='NETCDF4')
        lon=ncfile.variables['LON'][:]
        lat=ncfile.variables['LAT'][:]
        aice=ncfile.variables['AICE'][:]  #Ice concentration of grid cell
        hice=ncfile.variables['HICE'][:] #Mean ice thickness of grid cell
        fro=ncfile.variables['FROCEAN'][:]
        tmask=ncfile.variables['TMASK'][:] #Ocean mask (0=land, 1=ocean)
        area=ncfile.variables['AREA'][:]  #Area of grid cell (m2)
        ncfile.close()

        aicem=ma.masked_where(tmask[0,:,:]<0.5, aice[0,:,:])
        aicem=ma.masked_where(aicem<0.15, aicem)
        aicem=ma.masked_where(aicem>1.e10, aicem)
        aicem=ma.masked_where(hice[0,:,:]<0.06,  aicem)

        if POLE=='N':
            aicem=ma.masked_where(lat<0.0, aicem)
        if POLE=='S':
            aicem=ma.masked_where(lat>0.0, aicem)

        aicem_bool = ma.filled(aicem, fill_value=0)
        aicem_bool = np.where(aicem_bool>0,1,0)
        area_km = area[0,:,:]/(1000)**2 #m2 to km2
        extent[mon-1]=np.sum(aicem_bool*area_km)
    extent = extent*1.e-6
    ax[k].plot(months,extent, label='GEOS-S2S-3 climatology',c='b')


    # Set plotting settings
    ax[k].set_xticks(months)
    ax[k].set_xticklabels(monthlabel)
    ax[k].grid(True)
    ax[k].set_title(f'{POLE} Hemisphere',fontsize=16)
    # ax[k].set_ylim([5,22])

ax[0].set_ylabel('Sea Ice Extent ($10^{6}$ $km^{2}$)',fontsize=14)
# ax[0].legend(loc='upper center', bbox_to_anchor=(0, 0), ncol=2)
handles, labels = plt.gca().get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',ncol=2)

# Save and show plot
fig.savefig(PLOT_PATH+'/'+pngname, dpi=fig.dpi, bbox_inches='tight')
# plt.show()
