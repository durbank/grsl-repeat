# Module containing custon stats functions and operations

# Requisite modules
import pandas as pd
import geopandas as gpd
import numpy as np
from scipy.spatial.distance import pdist

def vario(
    points_gdf, lag_size, 
    d_metric='euclidean', vars='all', 
    standardize=False, scale=True):
    """A function to calculate the experimental variogram for values associated with a geoDataFrame of points.

    Args:
        points_gdf (geopandas.geodataframe.GeoDataFrame): Location of points and values associated with those points to use for calculating distances and lagged semivariance.
        lag_size (int): The size of the lagging interval to use for binning semivariance values.
        d_metric (str, optional): The distance metric to use when calculating pairwise distances in the geoDataFrame. Defaults to 'euclidean'.
        vars (list of str, optional): The names of variables to use when calculating semivariance. Defaults to 'all'.
        scale (bool, optional): Whether to perform normalization (relative to max value) on semivariance values. Defaults to True.

    Returns:
        pandas.core.frame.DataFrame: The calculated semivariance values for each chosen input. Also includes the lag interval (index), the average separation distance (dist), and the number of paired points within each interval (cnt). 
    """

    # Make copy to preserve original input df
    points_gdf = points_gdf.copy()

    # Get column names if 'all' is selected for "vars"
    if vars == "all":
        vars = points_gdf.drop(
            columns='geometry').columns

    if standardize:
        for var in vars:
            points_gdf[var] = (
                (points_gdf[var] - points_gdf[var].mean()) / points_gdf[var].std() )

    # Check if supplied gpd is Points or not
    if (~points_gdf.geom_type.isin(['Point'])).sum():
        points_gdf['geometry'] = points_gdf.centroid

    # Extact trace coordinates and calculate pairwise distance
    locs_arr = np.array(
        [points_gdf.geometry.x, points_gdf.geometry.y]).T
    dist_arr = pdist(locs_arr, metric=d_metric)

    # Calculate the indices used for each pairwise calculation
    i_idx = np.empty(dist_arr.shape)
    j_idx = np.empty(dist_arr.shape)
    m = locs_arr.shape[0]
    for i in range(m):
        for j in range(m):
            if i < j < m:
                i_idx[m*i + j - ((i + 2)*(i + 1))//2] = i
                j_idx[m*i + j - ((i + 2)*(i + 1))//2] = j
    
    # Create dfs for paired-point values
    i_vals = points_gdf[vars].iloc[i_idx].reset_index(
        drop=True)
    j_vals = points_gdf[vars].iloc[j_idx].reset_index(
        drop=True)

    # Calculate squared difference bewteen variable values
    sqdiff_df = (i_vals - j_vals)**2
    sqdiff_df['dist'] = dist_arr

    # Create array of lag interval endpoints
    d_max = lag_size * (dist_arr.max() // lag_size + 1)
    lags = np.arange(0,d_max+1,lag_size)

    # Group variables based on lagged distance intervals
    df_groups = sqdiff_df.groupby(
        pd.cut(sqdiff_df['dist'], lags))

    # Calculate semivariance at each lag for each variable
    gamma_vals = (1/2)*df_groups[vars].mean()
    gamma_vals.index.name = 'lag'

    if scale:
        gamma_df = gamma_vals / gamma_vals.max()
    else:
        gamma_df = gamma_vals

    # Add distance, lag center, and count values to output
    gamma_df['dist'] = df_groups['dist'].mean()
    gamma_df['lag_cent'] = lags[1::]-lag_size//2
    gamma_df['cnt'] = df_groups['dist'].count()

    return gamma_df

# Experiments with unit testing (currently does nothing)
if (__name__ == '__main__'):
    # import sys
    print('YAY FOR UNIT TESTING')