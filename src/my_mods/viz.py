# Module for project-specific visualizations and figure generation

# Requisite modules
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import holoviews as hv


def plot_TScomp(
    ts_df1, ts_df2, gdf_combo, labels, ts_cores=None,
    yaxis=True, xlims=None, ylims=None,
    colors=['blue', 'red'], ts_err1=None, ts_err2=None):
    """This is a function to generate matplotlib objects that compare spatially overlapping accumulation time series.

    Args:
        ts_df1 (pandas.DataFrame): Dataframe containing time series for the first dataset.
        ts_df2 (pandas.DataFrame): Dataframe containing time series for the second dataset.
        gdf_combo (geopandas.geoDataFrame): Geodataframe with entries corresponding to the paired time series locations. Also contains a column 'Site' that groups the different time series according to their manual tracing location.
        labels (list of str): The labels used in the output plot to differentiate the time series dataframes.
        colors (list, optional): The colors to use when plotting the time series. Defaults to ['blue', 'red'].
        ts_err1 (pandas.DataFrame, optional): DataFrame containing time series errors corresponding to ts_df1. If "None" then the error is estimated from the standard deviations in annual results. Defaults to None.
        ts_err2 (pandas.DataFrame, optional): DataFrame containing time series errors corresponding to ts_df2. If "None" then the error is estimated from the standard deviations in annual results. Defaults to None.

    Returns:
        matplotlib.pyplot.figure: Generated figure comparing the two overlapping time series.
    """

    # Remove observations without an assigned site
    site_list = np.unique(gdf_combo['Site']).tolist()
    if "Null" in site_list:
        site_list.remove("Null")

    # Generate figure with a row for each site
    fig, axes = plt.subplots(
        ncols=1, nrows=len(site_list), 
        constrained_layout=True, 
        figsize=(6,24))

    for i, site in enumerate(site_list):
        
        # Subset results to specific site
        idx = np.flatnonzero(gdf_combo['Site']==site)
        df1 = ts_df1.iloc[:,idx]
        df2 = ts_df2.iloc[:,idx]

        # Check if core data is provided and plot if available
        if ts_cores is not None:
            df_core = ts_cores.loc[:,site]
            df_core.plot(
                ax=fig.axes[i], color='black', linewidth=4, 
                linestyle=':', label=labels[2])

        # Check if errors for time series 1 are provided
        if ts_err1 is not None:

            df_err1 = ts_err1.iloc[:,idx]

            # Plot ts1 and mean errors
            df1.mean(axis=1).plot(ax=axes[i], color=colors[0], 
                linewidth=2, label=labels[0])
            (df1.mean(axis=1)+df_err1.mean(axis=1)).plot(
                ax=axes[i], color=colors[0], linestyle='--', 
                label='__nolegend__')
            (df1.mean(axis=1)-df_err1.mean(axis=1)).plot(
                ax=axes[i], color=colors[0], linestyle='--', 
                label='__nolegend__')
        else:
            # If ts1 errors are not given, estimate as the 
            # standard deviation of annual estimates
            df1.mean(axis=1).plot(ax=axes[i], color=colors[0], 
                linewidth=2, label=labels[0])
            (df1.mean(axis=1)+df1.std(axis=1)).plot(
                ax=axes[i], color=colors[0], linestyle='--', 
                label='__nolegend__')
            (df1.mean(axis=1)-df1.std(axis=1)).plot(
                ax=axes[i], color=colors[0], linestyle='--', 
                label='__nolegend__')

        # Check if errors for time series 2 are provided
        if ts_err2 is not None:
            df_err2 = ts_err2.iloc[:,idx]

            # Plot ts2 and mean errors
            df2.mean(axis=1).plot(
                ax=axes[i], color=colors[1], linewidth=2, 
                label=labels[1])
            (df2.mean(axis=1)+df_err2.mean(axis=1)).plot(
                ax=axes[i], color=colors[1], linestyle='--', 
                label='__nolegend__')
            (df2.mean(axis=1)-df_err2.mean(axis=1)).plot(
                ax=axes[i], color=colors[1], linestyle='--', 
                label='__nolegend__')
        else:
            # If ts2 errors are not given, estimate as the 
            # standard deviation of annual estimates
            df2.mean(axis=1).plot(
                ax=axes[i], color=colors[1], linewidth=2, 
                label=labels[1])
            (df2.mean(axis=1)+df2.std(axis=1)).plot(
                ax=axes[i], color=colors[1], linestyle='--', 
                label='__nolegend__')
            (df2.mean(axis=1)-df2.std(axis=1)).plot(
                ax=axes[i], color=colors[1], linestyle='--', 
                label='__nolegend__')
        
        if xlims:
            axes[i].set_xlim(xlims)
        if ylims:
            axes[i].set_ylim(ylims)

        # Add legend and set title based on site name
        axes[i].grid(True)

        if i == 0:
            axes[i].legend()
        # axes[i].set_title('Site '+site+' time series')

        if not yaxis:
            axes[i].set_yticklabels([])
        else:
            axes[i].set_ylabel('Accum (mm/a)')

        if i==(len(site_list)-1):
            pass
        else:
            axes[i].set_xticklabels([])
            axes[i].set_xlabel(None)

    return fig

def panels_121(
    datum, 
    x_vars=[
        'accum_2011','accum_2011','accum_man','accum_man'],
    y_vars=[
        'accum_2016','accum_2016','accum_paipr','accum_paipr'],
    TOP=False, BOTTOM=False, xlabels=None, ylabels=None, 
    kde_colors=['#d55e00', '#cc79a7', '#0072b2', '#009e73'],
    kde_labels=[
        '2016-2011 PAIPR', '2016-2011 manual', 
        '2011 PAIPR-manual', '2016 PAIPR-manual'], 
    plot_min=None, plot_max=None, size=500):
    """A function to generate a Holoviews Layout consisting of 1:1 plots of the given datum, with a kernel density plot also showing the residuals between the given x and y variables.

    Args:
        datum (list of pandas.DataFrame): First data group to include for generating plots. Variables in dataframe must include an x-variable, a y-variable, and a 'Year' variable.
        x_vars (list of str, optional): The names of the x-variables included in the datum. Defaults to [ 'accum_2011','accum_2011','accum_man','accum_man'].
        y_vars (list of str, optional): The names of the y-variables included in the datum.. Defaults to [ 'accum_2016','accum_2016','accum_paipr','accum_paipr'].
        TOP (bool, optional): Whether the returned Layout will be at the top of a Layout stack. Defaults to False.
        BOTTOM (bool, optional): Whether the returned Layout will be at the bottom of a Layout stack. Defaults to False.
        xlabels ([type], optional): List of strings to use as the x-labels for the 1:1 plots. Defaults to None.
        ylabels (list, optional): List of strings to use as the y-labels for the 1:1 plots. Defaults to None.
        kde_colors (list, optional): List of colors to use in kde plots. Defaults to ['blue', 'red', 'purple', 'orange'].
        kde_labels (list, optional): List of labels to use in kde plots. Defaults to [ '2016-2011 PAIPR', '2016-2011 manual', '2011 PAIPR-manual', '2016 PAIPR-manual'].
        plot_min (float, optional): Specify a lower bound to the generated plots. Defaults to None.
        plot_max (float, optional): Specifies an upper bound to the generated plots. Defaults to None.
        size (int, optional): The output size (both width and height) in pixels of individual subplots in the Layout panel. Defaults to 500.
        # font_scaler (float, optional): How much to scale the generated plot text. Defaults to no additional scaling.

    Returns:
        holoviews.core.layout.Layout: Figure Layout consisting of multiple 1:1 subplots and a subplot with kernel density estimates of the residuals of the various datum.
    """

    # Preallocate subplot lists
    one2one_plts = []
    kde_plots = []
    if xlabels is None:
        xlabels = np.repeat(None, len(datum))
    if ylabels is None:
        ylabels = np.repeat(None, len(datum))

    for i, data in enumerate(datum):

        # Get names of x-y variables for plotting
        x_var = x_vars[i]
        y_var = y_vars[i]

        # Get global axis bounds and generate 1:1 line
        if plot_min is None:
            plot_min = np.min(
                [data[x_var].min(), data[y_var].min()])
        if plot_max is None:
            plot_max = np.max(
                [data[x_var].max(), data[y_var].max()])
        one2one_line = hv.Curve(data=pd.DataFrame(
            {'x':[plot_min, plot_max], 
            'y':[plot_min, plot_max]})).opts(color='black')

        # Generate 1:1 scatter plot, gradated by year
        scatt_yr = hv.Points(
            data=data, kdims=[x_var, y_var], 
            vdims=['Year']).opts(
                xlim=(plot_min, plot_max), 
                ylim=(plot_min, plot_max), 
                xlabel=xlabels[i], 
                ylabel=ylabels[i], 
                color='Year', cmap='Category20b', colorbar=False)
        scatt_yr.opts(
            labelled=[], show_grid=True, xaxis=None, yaxis=None)

        # Special formatting given position of subplot in figure
        if i==0:
            scatt_yr.opts(yaxis='left')
            # scatt_yr.opts(ylabel='Annual accum (mm/yr)')
        if i==len(datum)-1:
            scatt_yr.opts(colorbar=True)
        if TOP:
            scatt_yr.opts(xaxis='top')
        if BOTTOM:
            scatt_yr.opts(xaxis='bottom')

        # Combine 1:1 line and scatter plot and add to plot list
        one2one_plt = (one2one_line * scatt_yr).opts(
            width=size, height=size, 
            # fontscale=3,
            fontsize={'ticks':20, 'xticks':30, 'yticks':30}, 
            xrotation=90, 
            xticks=4, yticks=4)
        if i==len(datum)-1:
            one2one_plt.opts(width=int(size+0.10*size))
        one2one_plts.append(one2one_plt)

        # Generate kde for residuals of given estimates and 
        # add to plot list 
        kde_data = (
            100*(data[y_var]-data[x_var]) 
            / data[[x_var, y_var]].mean(axis=1))
        kde_plot = hv.Distribution(
            kde_data, label=kde_labels[i]).opts(
            filled=False, line_color=kde_colors[i], 
            line_width=5)
        kde_plots.append(kde_plot)

    # Generate and decorate combined density subplot
    fig_kde = (
        kde_plots[0] * kde_plots[1] 
        * kde_plots[2] * kde_plots[3]).opts(
            xaxis=None, yaxis=None, 
            width=size, height=size, show_grid=True
        )
    
    # Specific formatting based on subplot position
    if TOP:
        fig_kde = fig_kde.opts(
            xaxis='top', xlabel='% Bias', xrotation=90, 
            fontsize={'legend':25, 'xticks':18, 'xlabel':22})
    elif BOTTOM:
        fig_kde = fig_kde.opts(
            xaxis='bottom', xlabel='% Bias', xrotation=90,
            fontsize={'legend':25, 'xticks':18, 'xlabel':22})
    else:
        fig_kde = fig_kde.opts(show_legend=False)


    # Generate final panel Layout with 1:1 and kde plots
    fig_layout = hv.Layout(
        one2one_plts[0] + one2one_plts[1] 
        + one2one_plts[2] + one2one_plts[3] 
        + fig_kde).cols(5)
    
    return fig_layout
