#! /usr/bin/env python
# This code computes batch jobs/large preprocessing of experiments for 
# further analysis.

import numpy as np
import numpy.ma as ma
import pandas as pd
# pd.options.mode.chained_assignment = None  # default='warn'
import xarray as xr
# xr.set_options(display_width=95, display_max_rows=40, display_expand_attrs=False)
import dask 
dask.config.set({"array.slicing.split_large_chunks": True}) 
import glob
import time
import datetime as dt
import sys
import os

# Define variables
COLLECTION1='geosgcm_seaice'
COLLECTION2='geosgcm_surf'
COLLECTION3 = 'geosgcm_prog'
EXPDIR='/nobackup/hzafar/Ys2s3_base_9'
HOMDIR=EXPDIR
EXPID=EXPDIR.split('/')[-1]
HAD_fileloc = './ObservationData/HADISST2/'
MERRA_fileloc = './transfer/'
S2S_fileloc = './interm_data/'


# 1 ################# Base 9 preprocessing ##################
####### AICE & EXTENT ########
'''
# Load dataset with wanted variables
fnames=EXPDIR+'/'+COLLECTION1+'/'+EXPID+'.'+COLLECTION1+'.monthly.??????.nc4'
variables = ['LAT','LON','AICE', 'HICE','TMASK', 'AREA'] #define wanted variables
drop_vars = ['DAIDTD', 'DIVU', 'DRAFT', 'DRAFT0', 'DVIDTD', 'DVIRDGDT', 'FHOCN', 'FRESH', 'FROCEAN', 'FSALT', 'HICE0', 'HSNO', 'HSNO0', 'SHEAR', 'SLV', 'SSH', 'STRCORX', 'STRCORY', 'STRENGTH', 'STRINTX', 'STRINTY', 'STRTLTX', 'STRTLTY', 'TAUXBOT', 'TAUXI', 'TAUXIB', 'TAUXOCNB', 'TAUYBOT', 'TAUYI', 'TAUYIB', 'TAUYOCNB','UI', 'UOCN', 'VEL', 'VI', 'VOCN', 'Var_AICE', 'Var_AREA', 'Var_DAIDTD', 'Var_DIVU', 'Var_DRAFT', 'Var_DRAFT0', 'Var_DVIDTD', 'Var_DVIRDGDT', 'Var_FHOCN', 'Var_FRESH', 'Var_FROCEAN', 'Var_FSALT', 'Var_HICE', 'Var_HICE0', 'Var_HSNO', 'Var_HSNO0', 'Var_SHEAR', 'Var_SLV', 'Var_SSH', 'Var_STRCORX', 'Var_STRCORY', 'Var_STRENGTH', 'Var_STRINTX', 'Var_STRINTY', 'Var_STRTLTX', 'Var_STRTLTY', 'Var_TAUXBOT', 'Var_TAUXI', 'Var_TAUXIB', 'Var_TAUXOCNB', 'Var_TAUYBOT', 'Var_TAUYI', 'Var_TAUYIB', 'Var_TAUYOCNB', 'Var_TMASK', 'Var_UI', 'Var_UOCN', 'Var_VEL', 'Var_VI', 'Var_VOCN']
# ds_s2s = xr.open_mfdataset(fnames, chunks='auto', drop_variables=drop_vars)
ds_s2s = xr.open_mfdataset(fnames, drop_variables=drop_vars)

#Replace tripolar lat/lon with LAT/LON
LAT = ds_s2s['LAT'][0,:,0].values
LON = ds_s2s['LON'][0,0,:].values
# print(LAT.shape,LON.shape,sep='\n')
ds_s2s.coords['lat'] = LAT
ds_s2s.coords['lon'] = LON

#Remove LAT/LON from variables
variables = ['HICE','AICE','TMASK','AREA']
ds_s2s = ds_s2s[variables]

#Clean Data
ds_s2s = ds_s2s.where((ds_s2s.TMASK>0.5) & (ds_s2s.HICE>0.06) & 
    (ds_s2s.AICE>0.15) & (ds_s2s.AICE<1.e10),drop=True)

#Compute extent for each pole
ds_s2s = ds_s2s.fillna(0)
ds_bool = ds_s2s['AICE'].where(ds_s2s.AICE==0,other=1).rename('AICE_bool')
# ds_merge = xr.merge([ds_bool, ds_s2s['AREA']])
ds_out = xr.Dataset()
variables_names = ['Extent_N', 'Extent_S']
POLES = ['N', 'S']
for vname, POLE in zip(variables_names, POLES):
    if POLE == 'N':
        ds_pole = ds_bool.where(ds_bool.lat>=30)
    else:
        ds_pole = ds_bool.where(ds_bool.lat<=-30)
    ds_out[vname] = ds_pole*ds_s2s['AREA']/(1000)**2 #convert m2 to km2

ds_extent = ds_out.sum(dim=['lat','lon'])[variables_names]
ds_extent = ds_extent.to_dataframe()

# Write data to netcdf for faster load later
#ds_s2s.to_netcdf('interm_data/preprocessed_seaice.nc')
ds_extent.to_csv('interm_data/preprocessed_seaice_extent.csv')
'''
######## SLP ########
'''
# Load dataset with wanted variables
state_var = ['SLP']
fnames = EXPDIR+'/'+COLLECTION3+'/'+EXPID+'.' +COLLECTION3+'.monthly.??????.nc4'
ds_s2s_state = xr.open_mfdataset(fnames)[state_var]

# Compute average SLP for each pole
# ds_state_out = ds_s2s_state.copy()
ds_state_out = xr.Dataset()

variables_names = ['SLP_N', 'SLP_S']
POLES = ['N', 'S']
for vname, POLE in zip(variables_names, POLES):
    if POLE == 'N':
        ds_pole = ds_s2s_state.where(ds_s2s_state.lat>=60)
    else:
        ds_pole = ds_s2s_state.where(ds_s2s_state.lat<=-60)
    ds_state_out[vname] = ds_pole['SLP']

print(ds_state_out)

# Write data to netcdf for faster load later
ds_name = 'preprocessed_' + state_var[0] + '.nc'
ds_state_out.to_netcdf('interm_data/'+ds_name)
sys.exit()
# Export as csv
ds_state_out = ds_state_out.mean(dim=["lat","lon"])
ds_state_out = ds_state_out.to_dataframe()
ds_state_out.to_csv('interm_data/preprocessed_SLP.csv')
'''
######## SNO (Snowfall) ########
'''
# Load dataset with wanted variables
state_var = ['SNO']
fnames = EXPDIR+'/'+COLLECTION2+'/'+EXPID+'.' +COLLECTION2+'.monthly.??????.nc4'
ds_s2s_state = xr.open_mfdataset(fnames)[state_var]

# Compute average SNO for each pole
ds_state_out = xr.Dataset()
variables_names = ['SNO_N', 'SNO_S']
POLES = ['N', 'S']
for vname, POLE in zip(variables_names, POLES):
    if POLE == 'N':
        ds_pole = ds_s2s_state.where(ds_s2s_state.lat>=60)
    else:
        ds_pole = ds_s2s_state.where(ds_s2s_state.lat<=-60)
    ds_state_out[vname] = ds_pole['SNO']

# Write data to netcdf for faster load later
print(ds_state_out)
ds_name = 'preprocessed_' + state_var[0] + '.nc'
ds_state_out.to_netcdf('interm_data/'+ds_name)

# Export as csv
ds_state_out = ds_state_out.mean(dim=["lat","lon"])
ds_state_out = ds_state_out.to_dataframe()
ds_state_out.to_csv('interm_data/preprocessed_SNO.csv')
print('done')
'''

######## TS (Skin Temp) ########
'''
# Load dataset with wanted variables
state_var = ['TS']
fnames = EXPDIR+'/'+COLLECTION2+'/'+EXPID+'.' +COLLECTION2+'.monthly.??????.nc4'
ds_s2s_state = xr.open_mfdataset(fnames)[state_var]

# Compute average TS for each pole
ds_state_out = ds_s2s_state.copy()
variables_names = ['TS_N', 'TS_S']
POLES = ['N', 'S']
for vname, POLE in zip(variables_names, POLES):
    if POLE == 'N':
        ds_pole = ds_s2s_state.where(ds_s2s_state.lat>=60)
    else:
        ds_pole = ds_s2s_state.where(ds_s2s_state.lat<=-60)
    ds_state_out[vname] = ds_pole['TS']

ds_state_out = ds_state_out.mean(dim=["lat","lon"])
ds_state_out = ds_state_out.to_dataframe()
ds_state_out.index = ds_state_out.index.normalize()

# Write data to netcdf for faster load later
ds_state_out.to_csv('interm_data/preprocessed_TS.csv')
print('done')
'''

# 2 ################# Scenario Processing ##################

######## Seasonal AICE ########
# '''
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

# Define misc variables
scenarios = ['High', 'Low', 'Mean']
df1 = pd.DataFrame(columns=['month','mean_MERRA','std_MERRA','mean_S2S','std_S2S'])
df2 = pd.DataFrame(columns=['month','Scenario','Count_MERRA', 'Count_S2S'])
df2 = pd.DataFrame(columns=['month','Scenario','Count_MERRA', 'Count_S2S'])
data_arrays = []

if POLE == 'N': 
  SEASON=['M03', 'M09']
  MONTH=['MAR', 'SEP']
  
else:
  SEASON=['M09', 'M02']
  MONTH=['SEP', 'FEB']

# Import HadISST2 data
fname=HAD_fileloc + 'HadISST.2.2.0.0_sea_ice_concentration.nc'
variable = ['sic']
ds_Had = xr.open_dataset(fname)[variable]
ds_Had = ds_Had.rename({'latitude': 'lat', 'longitude':'lon'}) #rename to match S2S
ds_Had.coords['lon'] = (ds_Had.coords['lon'] + 180) % 360 - 180 #convert to -180 to 180
ds_Had = ds_Had.sortby(ds_Had.lon)
ds_Had_selyr = ds_Had.sel(time=slice("1990-01-01", "2010-12-31")) #select 1990-2010

# Import MERRA2 analysis from discover (to classify HadISST)
fname = MERRA_fileloc + 'MERRA2_extract_' + statevar2 + '_'+POLE+'.csv'
ds_MERRA = pd.read_csv(fname,index_col=0, parse_dates=True)[statevar2]
# print(ds_MERRA)

# Import S2S Preprocessed state variable analysis data, drop 2015-2019 for model 
# initialization, & select hemisphere of interest
fname = S2S_fileloc + 'preprocessed_' + statevar + '.csv'
ds_s2s_state = pd.read_csv(fname,index_col=0, parse_dates=True)
ds_s2s_state = ds_s2s_state[ds_s2s_state.index.year>2019]
ds_s2s_state = ds_s2s_state[statevar+'_'+POLE]

# Convert from kg/m2-s to mm/day
if VAR == 'SNO': 
 ds_s2s_state = ds_s2s_state*86400
 ds_MERRA = ds_MERRA*86400

for i, (sea,mon_name) in enumerate(zip(SEASON,MONTH)):
  # Select month data
  mon = int(sea[-2:])

  ## HadISST2  & MERRA2 
  ds_Had_selmon = ds_Had_selyr.sel(time=ds_Had_selyr.time.dt.month == mon)
  ds_MERRA_selmon = ds_MERRA.loc[ds_MERRA.index.month == mon]

  ## S2S
  variables = ['FRSEAICE','FROCEAN'] #define wanted variables
  fname = EXPDIR+'/'+COLLECTION2+'/'+EXPID+'.'+COLLECTION2+ \
    '.monthly.????'+sea[-2:] +'.nc4'
  ds_s2s = xr.open_mfdataset(fname)[variables]
  ### Drop S2S 2015-2019 for model initialization:
  ds_s2s = ds_s2s.where(ds_s2s.time.dt.year>2019,drop=True)
  
  # Drop unneeded points for speed
  if POLE == 'N':
    ds_s2s = ds_s2s.where(ds_s2s.lat>=30,drop=True)
    ds_Had_selmon = ds_Had_selmon.where(ds_Had.lat>=30,drop=True)
  else:
    ds_s2s = ds_s2s.where(ds_s2s.lat<=-30,drop=True)
    ds_Had_selmon = ds_Had_selmon.where(ds_Had_selyr.lat<=-30,drop=True)
  
  ### Select S2S state for month
  ds_s2s_state_sel = ds_s2s_state.loc[ds_s2s_state.index.month == mon]

  # State variable analysis
  ## Compute means and std for state variables for MERRA2 and S2S
  mean_MER = ds_MERRA_selmon.mean()
  std_MER = ds_MERRA_selmon.std()
  mean_s2s = ds_s2s_state_sel.mean()
  std_s2s = ds_s2s_state_sel.std()

  # Explore data distribution
  # std_MER_true = std_MER
  # std_s2s_true = std_s2s
  # std_MER = 0.5*std_MER
  # std_s2s = 0.5*std_s2s
  # print(mon)
  # print('MERRA2:', mean_MER, std_MER_true, std_MER, sep='\t')
  # print('S2S:', mean_s2s, std_s2s_true, std_s2s, sep='\t')
  # ds_s2s_state_sel.hist()
  # ds_MERRA_selmon.hist()
  # vlines1 = [mean_MER,mean_MER-std_MER,mean_MER+std_MER]
  # plt.vlines(vlines1,0,5,color='orange',linestyle='dashed')
  # vlines2 = [mean_s2s,mean_s2s-std_s2s,mean_s2s+std_s2s]
  # plt.vlines(vlines2,0,40,color='c',linestyle='dashed')
  # plt.show()
  # sys.exit()

  df1.loc[i,['month']] = mon
  df1.loc[i,['mean_S2S']] = mean_s2s
  df1.loc[i,['std_S2S']] = std_s2s
  df1.loc[i,['mean_MERRA']] = mean_MER
  df1.loc[i,['std_MERRA']] = std_MER

  
  ## Create empty dataset to store data
  ds_interm = xr.Dataset()

  ## Extract time arrays for high/low/mean scenarios
  for j, scenario in enumerate(scenarios):

    times_MER_file = 'interm_data/times_data/'+'timesMER' + '_' + statevar + '_' + POLE  + '_'+ mon_name + '_' + scenario + '.npy'
    times_s2s_file = 'interm_data/times_data/'+'timess2s' + '_' + statevar + '_' + POLE  + '_'+ mon_name + '_' + scenario + '.npy'
    
    if scenario == 'High':
        times_MER = ds_MERRA_selmon.loc[ds_MERRA_selmon > (mean_MER+std_MER)].index
        times_s2s = ds_s2s_state_sel.loc[ds_s2s_state_sel > (mean_s2s+std_s2s)].index

        # print('# of ' + scenario, len(times_MER), len(times_s2s), sep='\t\t')
    if scenario == 'Low':
        times_MER = ds_MERRA_selmon.loc[ds_MERRA_selmon < (mean_MER-std_MER)].index
        times_s2s = ds_s2s_state_sel.loc[ds_s2s_state_sel < (mean_s2s-std_s2s)].index
        # print('# of ' + scenario, len(times_MER), len(times_s2s), sep='\t\t')
    if scenario == 'Mean':
        times_MER = ds_MERRA_selmon.loc[(ds_MERRA_selmon <= (mean_MER+std_MER))
            & (ds_MERRA_selmon >= (mean_MER-std_MER))].index
        times_s2s = ds_s2s_state_sel.loc[(ds_s2s_state_sel <= (mean_s2s+std_s2s))
            & (ds_s2s_state_sel >= (mean_s2s-std_s2s))].index
        # print('# of '+ scenario, len(times_MER), len(times_s2s), sep='\t\t')
    np.save(times_MER_file,times_MER)
    np.save(times_s2s_file,times_s2s)
  
    df2.loc[i*4+j,['month']] = mon
    df2.loc[i*4+j,['Scenario']] = scenario
    df2.loc[i*4+j,['Count_MERRA']] = len(times_MER)
    df2.loc[i*4+j,['Count_S2S']] = len(times_s2s)
    
    # Select HadISST & S2S data for the scenario and average
    ds_Had_scen = ds_Had_selmon.sel(time=times_MER, method='nearest').mean(dim='time')
    ds_s2s_scen = ds_s2s.sel(time=times_s2s, method='nearest').mean(dim='time')

    # Interpolate S2S to Had resolution
    ds_s2s_sceni = ds_s2s_scen.interp_like(ds_Had_scen)
    xr.align(ds_s2s_sceni, ds_Had_scen, join='exact') #Check coordinates match (gives error if not)

    # Compute AICE and Difference from Obs
    ds_s2s_sceni = ds_s2s_sceni.assign(AICE=ds_s2s_sceni['FRSEAICE']/ds_s2s_sceni['FROCEAN'])
    ds_s2s_sceni = ds_s2s_sceni.assign(DIFF_Had=ds_s2s_sceni['AICE'] - ds_Had_scen['sic'])

    # Export scenario to dataset
    S2S_AICE = scenario +'_AICE'
    S2S_FRSEAICE = scenario +'_FRSEAICE'
    S2S_FROCEAN = scenario +'_FROCEAN'
    Had_sic = scenario +'_sic'

    ds_interm[S2S_AICE] = ds_s2s_sceni['AICE']
    ds_interm[S2S_FRSEAICE] = ds_s2s_sceni['FRSEAICE']
    ds_interm[S2S_FROCEAN] = ds_s2s_sceni['FROCEAN']
    ds_interm[Had_sic] = ds_Had_scen['sic']

  ds_interm = ds_interm.assign_coords({'month': mon})
  ds_interm.load()
  # print(ds_interm)
  data_arrays.append(ds_interm)
  print(mon)
# sys.exit()

# Export netcdfs
ds = xr.concat(data_arrays,dim='month')
print(ds)
ds_name = 'preprocessed_aice_' + statevar + 'scenarios_seas_' +POLE + '.nc'
ds.to_netcdf('interm_data/'+ds_name)

# Export data for scenario categorization
## Assign names for files
df_name = statevar + 'scenarios_seas_' + POLE
df_path = 'interm_data/preprocessed_' + df_name + '.csv'

## Convert DataFrames to CSV format as strings
csv_string_df1 = df1.to_csv(index=False)
csv_string_df2 = df2.to_csv(index=False)

## Combine the CSV strings
combined_csv_string = df_name + '\n' + csv_string_df1 + '*********\n' + csv_string_df2

## Write the combined data to a new CSV file
with open(df_path, 'w') as combined_file:
    combined_file.write(combined_csv_string)
# '''

######## Full Year ########
'''
# THIS CODE NEEDS TO BE RERUN AS I UPDATED SCENARIO THRESHOLDS #####
# Running this code requires the following additional arguments: [{variable of interest}]['N' or 'S'] 
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

# Define scenarios
scenarios = ['High', 'Low', 'Mean']

# Define misc variables
months = np.arange(1,13)
data_arrays = []

# Import HadISST2 data
fname=HAD_fileloc + 'HadISST.2.2.0.0_sea_ice_concentration.nc'
variable = ['sic']
ds_Had = xr.open_dataset(fname)[variable]
ds_Had = ds_Had.rename({'latitude': 'lat', 'longitude':'lon'}) #rename to match S2S
ds_Had.coords['lon'] = (ds_Had.coords['lon'] + 180) % 360 - 180 #convert to -180 to 180
ds_Had = ds_Had.sortby(ds_Had.lon)
ds_Had_selyr = ds_Had.sel(time=slice("1990-01-01", "2010-12-31")) #select 1990-2010

# Import MERRA2 analysis from discover (to classify HadISST)
fname = MERRA_fileloc + 'MERRA2_extract_' + statevar2 + '_'+POLE+'.csv'
ds_MERRA = pd.read_csv(fname,index_col=0, parse_dates=True)[statevar2]

# Import S2S Preprocessed state variable analysis data (made from preprocessing above!)
# Drop 2015-2019 for model initialization & select hemisphere of interest
fname = S2S_fileloc + 'preprocessed_' + statevar + '.csv'
ds_s2s_state = pd.read_csv(fname,index_col=0, parse_dates=True)
ds_s2s_state = ds_s2s_state[ds_s2s_state.index.year>2019]
ds_s2s_state = ds_s2s_state[statevar+'_'+POLE]

# Convert from kg/m2-s to mm/day
if VAR == 'SNO': 
 ds_s2s_state = ds_s2s_state*86400
 ds_MERRA = ds_MERRA*86400

# Loop through months
for mon in months:
  sea = str(mon)
  if mon < 10:
      sea ='0'+str(mon)

  ## Select HadISST2  & MERRA2 for month
  ds_Had_selmon = ds_Had_selyr.sel(time=ds_Had_selyr.time.dt.month == mon)
  ds_MERRA_selmon = ds_MERRA.loc[ds_MERRA.index.month == mon]

  ## S2S
  ### Import S2S Seaice data for month of interest
  variables = ['FRSEAICE','FROCEAN'] #define wanted variables
  fname  = EXPDIR+'/'+COLLECTION2+'/'+EXPID \
    +'.'+COLLECTION2+'.monthly.????'+ sea +'.nc4'
  ds_s2s = xr.open_mfdataset(fname)[variables]
  
  ### Drop 2015-2019 for model initialization:
  ds_s2s = ds_s2s.where(ds_s2s.time.dt.year>2019,drop=True)
  
  # Drop unneeded points for speed
  if POLE == 'N':
    ds_s2s = ds_s2s.where(ds_s2s.lat>=30,drop=True)
    ds_Had_selmon = ds_Had_selmon.where(ds_Had.lat>=30,drop=True)
  else:
    ds_s2s = ds_s2s.where(ds_s2s.lat<=-30,drop=True)
    ds_Had_selmon = ds_Had_selmon.where(ds_Had_selyr.lat<=-30,drop=True)
  
  ### Select S2S state for month
  ds_s2s_state_sel = ds_s2s_state.loc[ds_s2s_state.index.month == mon]

  # State variable analysis
  ## Compute means and std for state variables for MERRA2 and S2S
  mean_MER = ds_MERRA_selmon.mean()
  std_MER = ds_MERRA_selmon.std()
  mean_s2s = ds_s2s_state_sel.mean()
  std_s2s = ds_s2s_state_sel.std()

  ## Extract time arrays for high/low/mean scenarios
  ds_interm = xr.Dataset()

  for j, scenario in enumerate(scenarios):
    if scenario == 'High':
        times_MER = ds_MERRA_selmon.loc[ds_MERRA_selmon > (mean_MER+std_MER)].index
        times_s2s = ds_s2s_state_sel.loc[ds_s2s_state_sel > (mean_s2s+std_s2s)].index
        # print('# of ' + scenario, len(times_MER), len(times_s2s), sep='\t\t')
    if scenario == 'Low':
        times_MER = ds_MERRA_selmon.loc[ds_MERRA_selmon < (mean_MER-std_MER)].index
        times_s2s = ds_s2s_state_sel.loc[ds_s2s_state_sel < (mean_s2s-std_s2s)].index
        # print('# of ' + scenario, len(times_MER), len(times_s2s), sep='\t\t')
    if scenario == 'Mean':
        times_MER = ds_MERRA_selmon.loc[(ds_MERRA_selmon <= (mean_MER+std_MER))
            & (ds_MERRA_selmon >= (mean_MER-std_MER))].index
        times_s2s = ds_s2s_state_sel.loc[(ds_s2s_state_sel <= (mean_s2s+std_s2s))
            & (ds_s2s_state_sel >= (mean_s2s-std_s2s))].index
        # print('# of '+ scenario, len(times_MER), len(times_s2s), sep='\t\t')
  
    ## Select HadISST & S2S data for the scenario and average
    ds_Had_scen = ds_Had_selmon.sel(time=times_MER, method='nearest').mean(dim='time')
    ds_s2s_scen = ds_s2s.sel(time=times_s2s, method='nearest').mean(dim='time')

    # Interpolate S2S to Had resolution
    ds_s2s_sceni = ds_s2s_scen.interp_like(ds_Had_scen)
    # xr.align(ds_s2s_sceni, ds_Had_scen, join='exact') #Check coordinates match (gives error if not)
    
    # Compute AICE at coarser reso
    ds_s2s_sceni = ds_s2s_sceni.assign(AICE=ds_s2s_sceni['FRSEAICE']/ds_s2s_sceni['FROCEAN'])

    # Export scenario to dataset
    S2S_AICE = scenario +'_AICE'
    S2S_FRSEAICE = scenario +'_FRSEAICE'
    S2S_FROCEAN = scenario +'_FROCEAN'
    Had_sic = scenario +'_sic'

    ds_interm[S2S_AICE] = ds_s2s_sceni['AICE']
    ds_interm[S2S_FRSEAICE] = ds_s2s_sceni['FRSEAICE']
    ds_interm[S2S_FROCEAN] = ds_s2s_sceni['FROCEAN']
    ds_interm[Had_sic] = ds_Had_scen['sic']

  ds_interm = ds_interm.assign_coords({'month': mon})
  ds_interm.load()
  # print(ds_interm)
  data_arrays.append(ds_interm)
  print(mon)

ds = xr.concat(data_arrays,dim='month')
print(ds)
ds_name = 'preprocessed_aice_diff_' + statevar + '_' +POLE + '.nc'
ds.to_netcdf('interm_data/'+ds_name)
'''