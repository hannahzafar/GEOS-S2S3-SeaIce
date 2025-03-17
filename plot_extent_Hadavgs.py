#!/usr/bin/env python
# This code generates Ice Extent time series plots for the annual cycle of sea ice, 
# comparing averages across 2, 5, and 10 year spans before and after yr 2000 for HadISST 
# Running this code requires no arguments.

from netCDF4 import Dataset,num2date,date2num
import numpy as np
import numpy.ma as ma
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.ticker import MaxNLocator
import xarray as xr
import pandas as pd
import glob
import datetime
import time
import sys
import os
import re

# Define input/output locations
COLLECTION='geosgcm_seaice'
EXPDIR='/nobackup/hzafar/Ys2s3_base_9'
# EXPDIR= #sys.argv[1] #extract experiment location from arguments
HOMDIR=EXPDIR
EXPID=EXPDIR.split('/')[-1]
PLOT_PATH = './plots/'
pngname = EXPID+'_ext_Hadavgs'

#Define arrays spanning 2000 to test
years =['1990-2010','1995-2005','1998-2002']
start_year = np.array([1990,1995,1998],dtype=int)
end_year = np.array([2010,2005,2002],dtype=int)

# Create figure
fig, ax = plt.subplots(1,2, figsize=(14,6))

# Define misc variables
months = np.arange(1,13,1)
x_axis = np.zeros(12)
for j in months:
    x_axis[j-1]=mdates.date2num(datetime.date(1900,j,1)) 

for k,POLE in enumerate(['N', 'S']):

    # Extract/modify/plot HADISST txt files
    fname=glob.glob('./ObservationData/HADISST2/HadISST.2.2.1.0_'+POLE+'*_'+'sea_ice_extent.txt')
    if POLE == 'S':
        ds_Had = pd.read_csv(fname[0],delim_whitespace=True, skiprows=1)
    else: 
        ds_Had = pd.read_csv(fname[0],delim_whitespace=True)
    ds_Had.columns = ['Year', 'Month', 'Extent']
    i=0
    for startyr, endyr in zip(start_year,end_year):
        start_date=str(startyr)+"-01-01"
        end_date=str(endyr)+"-12-31"
        ds_sel_yr = ds_Had[ds_Had['Year'].between(startyr, endyr)]
        ds_mon_avg = ds_sel_yr.groupby('Month')['Extent'].mean()
        ax[k].plot(x_axis,ds_mon_avg.values,label=f'HadISST2 ({years[i]})',linestyle='dashed',lw=2)
        i += 1

    # Extract/modify/plot S2S Experiment files
    extent=np.zeros(12)
    for mon in range(1,13,1):
        SEASON='M'+str(mon)
        if mon < 10:
            SEASON='M0'+str(mon)
        filename=EXPID+'.'+COLLECTION+'.monthly.clim.'+SEASON+'.nc4'
        fname=EXPDIR+'/'+COLLECTION+'/'+filename
        # print(fname)
        ncfile = Dataset(fname, 'r', format='NETCDF4')
        lon=ncfile.variables['LON'][:]
        lat=ncfile.variables['LAT'][:]
        aice=ncfile.variables['AICE'][:]  
        hice=ncfile.variables['HICE'][:] 
        fro=ncfile.variables['FROCEAN'][:]
        tmask=ncfile.variables['TMASK'][:] 
        area=ncfile.variables['AREA'][:] 
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
    ax[k].plot(x_axis,extent,label='GEOS-S2S-3 climatology',c='b',lw=3)

    # Set plotting settings
    ax[k].xaxis.set_major_locator(mdates.MonthLocator())
    ax[k].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax[k].yaxis.set_major_locator(MaxNLocator(integer=True))
    ax[k].set_ylabel('Extent ($10^{6}$ $km^{2}$)')
    ax[k].grid(True)
    ax[k].legend()
    ax[k].set_title(f'{POLE} Hemisphere')

# handles, labels = plt.gca().get_legend_handles_labels()
# fig.legend(handles,labels,loc='lower center',ncol=1)

# Save and show plot
fig.savefig(PLOT_PATH+'/'+pngname, dpi=fig.dpi, bbox_inches='tight')
# plt.show()
