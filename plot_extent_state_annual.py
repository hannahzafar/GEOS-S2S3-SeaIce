#!/usr/bin/env python
# This analyzes and plots sea ice extent for anomalous state parameters 
# (in the NH and SH compared to anomalous HadISST2.
# Running this code requires no additional arguments

import numpy as np
import numpy.ma as ma
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import xarray as xr
import glob
import datetime 
import time
import sys
import os
# import re
# import struct

# Define input/output locations
COLLECTION1 = 'geosgcm_seaice'
COLLECTION2 = 'geosgcm_surf'
COLLECTION3 = 'geosgcm_prog'
EXPDIR='/nobackup/hzafar/Ys2s3_base_9' 
# EXPDIR= #sys.argv[1] #extract experiment location from arguments
HOMDIR=EXPDIR
EXPID=EXPDIR.split('/')[-1]
PLOT_PATH = './plots/'
pngname = EXPID+'_ext_slpanom'
HAD_fileloc = './ObservationData/HADISST2/'
MERRA_fileloc = './transfer/'
S2S_fileloc = './interm_data/'

# Define years for HADISST2
startyr = 1990
endyr = 2010
start_date=str(startyr)+"-01-01"
end_date=str(endyr)+"-12-31"

#Define scenarios
scenarios = ['High SLP', 'Low SLP', 'Control SLP']
# Create figure/axes
fig, axes = plt.subplots(2,3, figsize=(16, 10), constrained_layout=True)
fig.suptitle('Sea Ice Extent ($10^{6}$ $km^{2}$)',fontsize=15)  
# Define misc variables
months = np.arange(1,13)
monthlabel = [datetime.date(1900, mon, 1).strftime('%b') for mon in months]
Had_ext_scenarios = np.zeros((3,12))
s2s_ext_scenarios = np.zeros((3,12))
s2s_std_scenarios = np.zeros((3,12))
Had_std_scenarios = np.zeros((3,12))

# x_axis = np.zeros(12)
# for j in months:
#     x_axis[j-1]=mdates.date2num(datetime.date(1900,j,1)) 

# Import S2S Preprocessed
## S2S sea ice extent
fname = S2S_fileloc + 'preprocessed_seaice_extent.csv'
ds_s2s = pd.read_csv(fname,index_col=0, parse_dates=True)
ds_s2s.index = ds_s2s.index.normalize() #Remove times from datetime
## S2S state variable analysis data
fname = S2S_fileloc + 'preprocessed_SLP.csv'
ds_s2s_state = pd.read_csv(fname,index_col=0, parse_dates=True)

# Loop over N and S hemispheres
for i,POLE in enumerate(['N', 'S']):
    
    # Extract/modify HADISST txt files
    fname= HAD_fileloc + 'HadISST.2.2.1.0_'+POLE+'H_sea_ice_extent.txt'
    if POLE == 'S':
        ds_Had = pd.read_csv(fname,delim_whitespace=True, skiprows=1)
    else: 
        ds_Had = pd.read_csv(fname,delim_whitespace=True)
    ds_Had.columns = ['Year', 'Month', 'Extent']
    ds_Had_sel = ds_Had[ds_Had['Year'].between(startyr, endyr)]
    ds_Had_sel['time'] = pd.to_datetime(dict(year= ds_Had_sel.Year, month= ds_Had_sel.Month, day=1))
    ds_Had_sel = ds_Had_sel.set_index(['time'])['Extent']

    # Import MERRA2 analysis from discover (to classify HadISST)
    fname = MERRA_fileloc + 'MERRA2_extract_SLP_'+POLE+'.csv'
    ds_MERRA = pd.read_csv(fname,index_col=0, parse_dates=True)['SLP']

    #Select hemisphere S2S data
    ds_s2sH = ds_s2s['Extent_'+POLE]
    ds_s2s_stateH = ds_s2s_state['SLP_'+POLE]

    # Loop through each month
    for mon in months:
        # Select month of interest
        ds_MERRA_selmon = ds_MERRA.loc[ds_MERRA.index.month == mon]
        ds_s2s_selmon = ds_s2s_stateH.loc[ds_s2s_stateH.index.month == mon]

        # State variable analysis
        ## Compute means and std for state variables for MERRA2 and S2S
        mean_MER = ds_MERRA_selmon.mean()
        std_MER = ds_MERRA_selmon.std()
        mean_s2s = ds_s2s_selmon.mean()
        std_s2s = ds_s2s_selmon.std()

        for j, scenario in enumerate(scenarios):
            if scenario == 'High SLP':
                times_MER = ds_MERRA_selmon.loc[ds_MERRA_selmon > (mean_MER+std_MER)].index
                times_s2s = ds_s2s_selmon.loc[ds_s2s_selmon > (mean_s2s+std_s2s)].index
                print(times_MER)
            if scenario == 'Low SLP':
                times_MER = ds_MERRA_selmon.loc[ds_MERRA_selmon < (mean_MER-std_MER)].index
                times_s2s = ds_s2s_selmon.loc[ds_s2s_selmon < (mean_s2s-std_s2s)].index
            
            if scenario == 'Control SLP':
                times_MER = ds_MERRA_selmon.loc[(ds_MERRA_selmon <= (mean_MER+std_MER))
                    & (ds_MERRA_selmon >= (mean_MER-std_MER))].index
                times_s2s = ds_s2s_selmon.loc[(ds_s2s_selmon <= (mean_s2s+std_s2s))
                    & (ds_s2s_selmon >= (mean_s2s-std_s2s))].index
            # print(times_MER)
            # print(times_s2s)
            ds_Had_scen = ds_Had_sel.loc[times_MER].mean()
            Had_ext_scenarios[j,mon-1] = ds_Had_scen 

            ds_Had_std = ds_Had_sel.loc[times_MER].std()
            Had_std_scenarios[j,mon-1] = ds_Had_std

            ds_s2s_scen = ds_s2sH.loc[times_s2s].mean()*1.e-6
            s2s_ext_scenarios[j,mon-1] = ds_s2s_scen

            ds_s2s_std = ds_s2sH.loc[times_s2s].std()*1.e-6
            s2s_std_scenarios[j,mon-1] = ds_s2s_std
            # print(ds_s2s_std)
            # sys.exit()


   
    for j, scenario in enumerate(scenarios):
        axes[i,j].plot(months, Had_ext_scenarios[j,:], c='m', linestyle='dashed',label='HadISST2')
        axes[i,j].plot(months, s2s_ext_scenarios[j,:], c='b', label='S2S Base9')
        # axes[i,j].errorbar(months, s2s_ext_scenarios[j,:],s2s_std_scenarios[j,:], label='S2S Base9')
        axes[i,j].fill_between(months,Had_ext_scenarios[j,:]-Had_std_scenarios[j,:],Had_ext_scenarios[j,:]+Had_std_scenarios[j,:],color='m',alpha=0.2)
        axes[i,j].fill_between(months,s2s_ext_scenarios[j,:]-s2s_std_scenarios[j,:],s2s_ext_scenarios[j,:]+s2s_std_scenarios[j,:],color='b', alpha=0.2)
        #Plot formatting
        axes[i,j].set_xticks(months)
        axes[i,j].set_xticklabels(monthlabel)
        axes[i,j].legend()
        axes[i,j].grid(True)
        # axes[i,j].set_ylabel('Extent ($10^{6}$ $km^{2}$)')
        pole = POLE +'H'
        axes[i,0].set_ylabel(pole,fontsize=12, rotation='horizontal',labelpad=40)
        axes[1,j].set_title('')
        axes[0,j].set_title(scenario,fontsize=18)
        # plt.show() 
        # sys.exit()
       
# Save and show plot
# fig.savefig(PLOT_PATH+'/'+pngname, dpi=fig.dpi, bbox_inches='tight')
# print(Had_std_scenarios)
# plt.show()
