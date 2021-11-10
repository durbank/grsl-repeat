# Module of helper functions for formatting, splitting, and otherwise dealing with PAIPR-generated results

from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np
import xarray as xr

def import_PAIPR(input_dir):
    """
    Function to import PAIPR-derived accumulation data.
    This concatenates all files within the given directory into a single pandas dataframe.

    Parameters:
    input_dir (pathlib.PosixPath): Absolute path of directory containing .csv files of PAIPR results. This directory can contain multiple files, which will be concatenated to a single dataframe.
    """

    data = pd.DataFrame()
    for file in input_dir.glob("*.csv"):
        data_f = pd.read_csv(file)
        data = data.append(data_f)
    data['collect_time'] = pd.to_datetime(
        data['collect_time'])
    return data

def format_PAIPR(
    data_df, start_yr=None, end_yr=None, 
    yr_clip=True, rm_deep=True):
    """

    """
    # Create groups based on trace locations
    traces = data_df.groupby(
        ['collect_time', 'Lat', 'Lon'])
    data_df = data_df.assign(trace_ID = traces.ngroup())
    traces = data_df.groupby('trace_ID')

    if start_yr != end_yr:
        # Remove time series with data missing from period
        # of interest (and clip to period of interest)
        data_df = traces.filter(
            lambda x: min(x['Year']) <= start_yr 
            and max(x['Year']) >= end_yr)
        data_df = data_df.query(
            f"Year >= {start_yr} & Year <= {end_yr}")

    # # Ensure each trace has only one time series 
    # # (if not, take the mean of all time series)
    # data = data.groupby(['trace_ID', 'Year']).mean()

    if 'gamma_shape' in data_df.columns:
        # Generate descriptive statistics based on 
        # imported gamma-fitted parameters
        alpha = data_df['gamma_shape']
        alpha.loc[alpha<1] = 1
        beta = 1/data_df['gamma_scale']
        mode_accum = (alpha-1)/beta
        var_accum = alpha/beta**2
        
        # New df (in long format) with accum data assigned
        data_long = (
            data_df.filter(['trace_ID', 'collect_time', 
            'QC_flag', 'Lat', 'Lon', 
            'elev', 'Year']).assign(
                accum=mode_accum, 
                std=np.sqrt(var_accum))
            .reset_index(drop=False))
    else:
        # New df (in long format) with accum data assigned
        data_long = (
            data_df.filter(['trace_ID', 'collect_time', 
            'QC_flag', 'QC_med', 'Lat', 'Lon', 
            'elev', 'Year']).assign(
                accum=data_df['accum_mu'], 
                std=data_df['accum_std']).reset_index(drop=True))

    if rm_deep:
        # Additional subroutine to remove time series where the deepest 3 years have overly large uncertainties (std > expected value)
        data_tmp = data_long.groupby(
            'trace_ID').mean().query('std < accum').reset_index()
        IDs_keep = data_tmp['trace_ID']
        data_long = data_long[
            data_long['trace_ID'].isin(IDs_keep)]

    if yr_clip:
        # Remove time series with fewer than 5 years
        data_tmp = data_long.join(
            data_long.groupby('trace_ID')['Year'].count(), 
            on='trace_ID', rsuffix='_count')
        data_long = data_tmp.query('Year_count >= 5').drop(
            'Year_count', axis=1)

    # Reset trace IDs to match total number of traces
    tmp_group = data_long.groupby('trace_ID')
    data_long['trace_ID'] = tmp_group.ngroup()
    data_final = data_long.sort_values(
        ['trace_ID', 'Year']).reset_index(drop=True)

    return data_final

def long2gdf(accum_long):
    """
    Function to convert data in long format to geodataframe aggregated
    by trace location and sorted by collection time
    """

    accum_long['collect_time'] = (
        accum_long.collect_time.values.astype(np.int64))
    traces = accum_long.groupby('trace_ID').mean().drop(
        'Year', axis=1)
    traces['collect_time'] = (
        pd.to_datetime(traces.collect_time)
        .dt.round('1ms'))

    # Sort by collect_time and reset trace_ID index
    # traces = (traces.sort_values('collect_time')
    #     .reset_index(drop=True))
    # traces.index.name = 'trace_ID'
    traces = traces.reset_index()

    gdf_traces = gpd.GeoDataFrame(
        traces.drop(['Lat', 'Lon'], axis=1), 
        geometry=gpd.points_from_xy(
            traces.Lon, traces.Lat), 
        crs="EPSG:4326")

    return gdf_traces

def get_FlightData(flight_dir):
    """Function to extract OIB flight parameter data from .nc files and convert to geodataframe.

    Args:
        flight_dir (pathlib.PosixPath): The directory containing the .nc OIB files to extract and convert.

    Returns:
        geopandas.geodataframe.GeoDataFrame: Table of OIB flight parameters (altitude, heading, lat, lon, pitch, and roll) with their corresponding surface location and collection time.
    """
    # List of files to extract from
    nc_files = [file for file in flight_dir.glob('*.nc')]
    
    # Load as xarray dataset
    xr_flight = xr.open_dataset(
        nc_files.pop(0))[[
            'altitude', 'heading', 'lat', 
            'lon', 'pitch','roll']]

    # Concatenate data from all flights in directory
    for file in nc_files:

        flight_i = xr.open_dataset(file)[[
            'altitude', 'heading', 'lat', 
            'lon', 'pitch','roll']]
        xr_flight = xr.concat(
            [xr_flight, flight_i], dim='time')

    # Convert to dataframe and average values to ~200 m resolution
    flight_data = xr_flight.to_dataframe()
    flight_coarse = flight_data.rolling(
        window=40, min_periods=1).mean().iloc[0::40]

    # Convert to geodataframe in Antarctic coordinates
    gdf_flight = gpd.GeoDataFrame(
        data=flight_coarse.drop(columns=['lat', 'lon']), 
        geometry=gpd.points_from_xy(
            flight_coarse.lon, flight_coarse.lat), 
        crs='EPSG:4326').reset_index().to_crs(epsg=3031)

    return gdf_flight

# Experiments with unit testing (currently does nothing)
if (__name__ == '__main__'):
    # import sys
    print('YAY FOR UNIT TESTING')
