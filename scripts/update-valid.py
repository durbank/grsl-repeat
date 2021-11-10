# An updated validation of the most recent PAIPR results using both firn cores and manual radar data.

# %% Set up environment

# Import requisite modules
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from cartopy import crs as ccrs
import holoviews as hv
hv.extension('bokeh', 'matplotlib')

# Define plotting projection to use
ANT_proj = ccrs.SouthPolarStereo(true_scale_latitude=-71)

# Set project root directory
ROOT_DIR = ROOT_DIR = Path(__file__).parents[1]

# Import custom project functions
import sys
SRC_DIR = ROOT_DIR.joinpath('src')
sys.path.append(str(SRC_DIR))
from my_mods import paipr, viz
from my_mods import spat_ops as so

# %% Import and format radar data

# Import and format PAIPR results
dir1 = ROOT_DIR.joinpath(
    'data/PAIPR-repeat/20111109/smb')
data_raw = paipr.import_PAIPR(dir1)
data_raw.query('QC_flag != 2', inplace=True)
data_0 = data_raw.query(
    'Year > QC_yr').sort_values(
    ['collect_time', 'Year']).reset_index(drop=True)
paipr_data = paipr.format_PAIPR(
    data_0, start_yr=1980, end_yr=2010).drop(
    'elev', axis=1)
paipr_ALL = paipr_data.pivot(
    index='Year', columns='trace_ID', values='accum')
std_ALL = paipr_data.pivot(
    index='Year', columns='trace_ID', values='std')

# Import and format manual results
dir_0 = ROOT_DIR.joinpath(
    'data/PAIPR-repeat/20111109/smb-manual/')
data_0 = paipr.import_PAIPR(dir_0)
man_data = paipr.format_PAIPR(
    data_0, start_yr=1980, end_yr=2010).drop(
    'elev', axis=1)
man_ALL = man_data.pivot(
    index='Year', columns='trace_ID', values='accum')
manSTD_ALL = man_data.pivot(
    index='Year', columns='trace_ID', values='std')

# Create gdf of mean results for each trace and 
# transform to Antarctic Polar Stereographic
gdf_paipr = paipr.long2gdf(paipr_data)
gdf_paipr.to_crs(epsg=3031, inplace=True)
gdf_man = paipr.long2gdf(man_data)
gdf_man.to_crs(epsg=3031, inplace=True)

# Drop unwanted columns
gdf_paipr.drop(
    ['collect_time', 'QC_flag'], axis=1, inplace=True)
gdf_man.drop(
    ['collect_time', 'QC_flag'], axis=1, inplace=True)

# %% Import and format core data

# Import raw data
samba_raw = pd.read_excel(ROOT_DIR.joinpath(
    "data/DGK_SMB_compilation.xlsx"), 
    sheet_name='Accumulation')

# Format SAMBA core data
core_ALL = samba_raw.iloc[3:,1:]
core_ALL.index = samba_raw.iloc[3:,0]
core_ALL.index.name = 'Year'
core_meta = samba_raw.iloc[0:3,1:]
core_meta.index = ['Lat', 'Lon', 'Elev']
core_meta.index.name = 'Attributes'
new_row = core_ALL.notna().sum()
new_row.name = 'Duration'
core_meta = core_meta.append(new_row)

# Flip to get results in tidy-compliant form
core_meta = core_meta.transpose()
core_meta.index.name = 'Name'

# Convert to geodf and reproject to 3031
core_locs = gpd.GeoDataFrame(
    data=core_meta.drop(['Lat', 'Lon'], axis=1), 
    geometry=gpd.points_from_xy(
        core_meta.Lon, core_meta.Lat), 
    crs='EPSG:4326')
core_locs.to_crs('EPSG:3031', inplace=True)

# %% Filter datum to desired subsets

# Filter results to desired cores
cores_keep = ['SEAT-10-4', 'SEAT-10-5', 'SEAT-10-6', 'PIG2010']
gdf_cores = core_locs.loc[cores_keep]
accum_cores = core_ALL.loc[:,cores_keep]
accum_cores.dropna(axis=0, how='all', inplace=True)

# Add site labels to paipr data
gdf_paipr['Site'] = np.repeat('Null', gdf_paipr.shape[0])
for site in gdf_cores.index:

    geom = gdf_cores.loc[site].geometry
    idx = (gpd.GeoSeries(
        data=gpd.points_from_xy(
            np.repeat(geom.x, gdf_paipr.shape[0]), 
            np.repeat(geom.y, gdf_paipr.shape[0])), 
        crs=gdf_paipr.crs).distance(
            gdf_paipr.reset_index()) <= 15000).values
    
    gdf_paipr.loc[idx,'Site'] = site

# Add site labels to manual data
gdf_man['Site'] = np.repeat('Null', gdf_man.shape[0])
for site in gdf_cores.index:

    geom = gdf_cores.loc[site].geometry
    idx = (gpd.GeoSeries(
        data=gpd.points_from_xy(
            np.repeat(geom.x, gdf_man.shape[0]), 
            np.repeat(geom.y, gdf_man.shape[0])), 
        crs=gdf_man.crs).distance(
            gdf_man.reset_index()) <= 15000).values
    
    gdf_man.loc[idx,'Site'] = site

# Filter radar results to within +/-15 km of a core
gdf_paipr.query('Site != "Null"', inplace=True)
gdf_man.query('Site != "Null"', inplace=True)

# %% Subset results to matched pairs

# Find nearest neighbors between paipr and manual 
# (within 250 m)
df_dist = so.nearest_neighbor(
    gdf_paipr, gdf_man, return_dist=True)
idx_paipr = df_dist['distance'] <= 250
dist_overlap = df_dist[idx_paipr]

# Create arrays for relevant results
accum_paipr = paipr_ALL.iloc[
    :,gdf_paipr.iloc[dist_overlap.index]['trace_ID']]
std_paipr = std_ALL.iloc[
    :,gdf_paipr.iloc[dist_overlap.index]['trace_ID']]
accum_man = man_ALL.iloc[:,dist_overlap['trace_ID']]
std_man = manSTD_ALL.iloc[:,dist_overlap['trace_ID']]

# Create new gdf of subsetted results
gdf_radar = gpd.GeoDataFrame(
    {'ID_paipr': gdf_paipr.iloc[
        dist_overlap.index]['trace_ID'].values, 
    'ID_man': dist_overlap['trace_ID'].values, 
    'accum_paipr': 
        accum_paipr.mean(axis=0).values, 
    'accum_man': 
        accum_man.mean(axis=0).values,
    'Site': dist_overlap.Site},
    geometry=dist_overlap.geometry.values)

# %%

tsfig = viz.plot_TScomp(
    ts_df1=accum_paipr, ts_df2=accum_man, 
    gdf_combo=gdf_radar, ts_cores=accum_cores,
    yaxis=True, xlims=[1980, 2010], ylims=[80, 700],
    labels=['PAIPR', 'Manual', 'Core'], 
    ts_err1=std_paipr, ts_err2=std_man)

# %% Calculate residuals between data

cores_rep = accum_cores.loc[
    2010:1980,gdf_radar['Site'].values].iloc[::-1]

PM_res = pd.DataFrame(
    accum_paipr.values-accum_man.values,
    index=accum_man.index)
PC_res = pd.DataFrame(
    100*(
        accum_paipr.values-cores_rep.values)/cores_rep.values, 
    index=cores_rep.index)
MC_res = pd.DataFrame(
    100*(accum_man.values-cores_rep.values)/cores_rep.values, 
    index=cores_rep.index)


print(f"Biases are as follows:")
print(f"PAIPR-manual: {PM_res.values.mean():.1f} mm/a ({100*(PM_res.values/accum_man.values).mean():.0f}%)")
print(f"PAIPR-core: {PC_res.values.mean():.1f} mm/a ({100*(PC_res.values/cores_rep.values).mean():.0f}%)")
print(f"Manual-core: {MC_res.values.mean():.1f} mm/a ({100*(MC_res.values/cores_rep.values).mean():.0f}%)")
print()
print(f"RMSE values are as follows:")
print(f"PAIPR-manual: {100*np.sqrt(((PM_res/accum_man.values)**2).values.mean()):.0f}%")
print(f"PAIPR-core: {100*np.sqrt(((PC_res/cores_rep.values)**2).values.mean()):.0f}%")
print(f"manual-core: {100*np.sqrt(((MC_res/cores_rep.values)**2).values.mean()):.0f}%")

# %%

# Create dataframes for scatter plots
df_radar = pd.DataFrame(
    {'tmp_ID': np.tile(
        np.arange(0,accum_paipr.shape[1]), 
        accum_paipr.shape[0]), 
    'Site': np.tile(
        gdf_radar['Site'], accum_paipr.shape[0]), 
    'Year': np.reshape(
        np.repeat(accum_paipr.index, accum_paipr.shape[1]), 
        accum_paipr.size), 
    'accum_paipr': 
        np.reshape(
            accum_paipr.values, accum_paipr.size), 
    'std_paipr': np.reshape(
        std_paipr.values, std_paipr.size), 
    'accum_man': np.reshape(
        accum_man.values, accum_man.size), 
    'std_man': np.reshape(
        std_man.values, std_man.size), 
    'accum_core': np.reshape(
        cores_rep.values, cores_rep.size)})



one2one = hv.Curve(
    data=pd.DataFrame(
        {'x':[100,750], 'y':[100,750]}))

scatter_0 = hv.Points(
    data=df_radar, 
    kdims=['accum_man', 'accum_paipr'], 
    vdims=['Year']).opts(
        xlim=(100,750), ylim=(100,750), 
        xlabel='manual accum (mm/yr)', 
        ylabel='paipr accum (mm/yr)', 
        color='Year', cmap='plasma', colorbar=True, 
        width=600, height=600, fontscale=1.75)
scatter_1 = hv.Points(
    data=df_radar, 
    kdims=['accum_core', 'accum_paipr'], 
    vdims=['Year']).opts(
        xlim=(100,750), ylim=(100,750), 
        xlabel='core accum (mm/yr)', 
        ylabel='paipr accum (mm/yr)', 
        color='Year', cmap='plasma', colorbar=True, 
        width=600, height=600, fontscale=1.75)
scatter_2 = hv.Points(
    data=df_radar, 
    kdims=['accum_core', 'accum_man'], 
    vdims=['Year']).opts(
        xlim=(100,750), ylim=(100,750), 
        xlabel='core accum (mm/yr)', 
        ylabel='manual accum (mm/yr)', 
        color='Year', cmap='plasma', colorbar=True, 
        width=600, height=600, fontscale=1.75)


one2one_plts = (
    one2one.opts(color='black')*scatter_0
    + one2one.opts(color='black')*scatter_1
    + one2one.opts(color='black')*scatter_2
)

one2one_plts


# %% Subset results to matched pairs for cores only

# Find nearest neighbors between paipr and manual 
# (within 500 m)
df_dist = so.nearest_neighbor(
    gdf_cores, gdf_paipr, return_dist=True)
idx_paipr = df_dist['distance'] <= 6000
dist_overlap = df_dist[idx_paipr]

# Create arrays for relevant results
accum_paipr = paipr_ALL.iloc[:,dist_overlap['trace_ID']]
std_paipr = std_ALL.iloc[:,dist_overlap['trace_ID']]



cores = accum_cores.loc[2010:1980].iloc[::-1]

paipr_res = pd.DataFrame(
    accum_paipr.values-cores.values,
    index=accum_paipr.index)

print(f"Mean PAIPR-core bias: {100*paipr_res.iloc[:,1::].values.mean()/accum_paipr.values.mean():.1f}%")
print(f"Mean PAIPR-core RMSE: {100*paipr_res.iloc[:,1::].values.std()/cores.values.mean():.0f}%")

# %%
