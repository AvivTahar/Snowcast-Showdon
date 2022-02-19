import pandas as pd
import numpy as np
from scipy.interpolate import Rbf

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io

from flask import Flask, request, Response
app = Flask(__name__)

RES_X = RES_Y = 750
SWE_COLS = 3

# plot random station URL:  http://localhost:105/plot_station/?station=random
# plot random date URL:     http://localhost:105/plot_date/?date=random


def get_train_metadata(input_path, metadata_path, key, drop_index=True, norm_cols=True):
    """
    returns input file joined with longitude and latitude metadata

    key:        input file column by which to perform the join
    drop_index: remove redundant index column from the input file
    norm_cols:  normalize date columns format

    return:     dataframe object with columns of this shape [key, longitude, latitude, [2020-21 date columns]]
    """
    # read files and rename cell id column of metadata dataframe to match with input files' same column using key
    metadata_df = pd.read_csv(metadata_path)[['field_1', 'longitude', 'latitude']].rename(columns={'field_1': key})
    input_df = pd.read_csv(input_path)
    # remove input dataframes' index if necessary
    if drop_index:
        input_df.drop(columns='Unnamed: 0', inplace=True)
    # merge both dataframes on cell id column, retain only rows that are in input file
    result = metadata_df.merge(input_df, how='right')
    # normalize date columns
    columns = result.columns
    result.columns = list(columns[:SWE_COLS]) + [col[:-9] for col in columns[SWE_COLS:]]

    return result


def generate_grid(long_co, lat_co):
    """
    return X and Y coordinates of encompassing grid of given longitudes and latitudes

    return: longitude values array, latitude values array
    """
    # create grid of all cells around max/min coordinates
    x_max = long_co.max()
    x_min = long_co.min()
    y_max = lat_co.max()
    y_min = lat_co.min()

    xi = np.linspace(x_min, x_max, RES_X)
    yi = np.linspace(y_min, y_max, RES_Y)
    xi, yi = np.meshgrid(xi, yi)
    # flatten the vectors of the grid for usage and invert the latitudes for better visualization
    xi, yi = xi.flatten(), yi.flatten()
    yi = yi[::-1]

    return xi, yi


def interpolate_snapshot(x, y, z, xi, yi):
    """
    interpolate SWE grid using linear rbf

    x, y, z: longitude, latitude, swe to interpolate from
    xi, yi:  longitude, latitude to interpolate at

    return:  SWE interpolated at (xi, yi) points
    """
    # interpolate using linear rbf
    interp = Rbf(x, y, z, function='linear')
    result = interp(xi, yi)
    # return result after clipping negative SWE predictions
    return np.clip(result, 0, None)


@app.route('/plot_station/', methods=['GET', 'POST'])
def plot_station():
    """
    returns png file with plot of SWE forecasts over 2020-21 snow seasons for a given station

    station: URL argument, name of station to plot. if 'random' chosen randomly
    """

    fig = plt.Figure(figsize=(10, 10))
    axis = fig.add_subplot(111)
    forecasts = pd.read_csv("forecasts.csv")
    # initialize station data according to URL argument
    station = request.args['station']
    if station == 'random':
        station_data = forecasts.sample(1)
        station = station_data['Station'].values[0]
    elif station not in forecasts['Station']:
        return f'station {station} not found'
    else:
        station_data = forecasts[forecasts['Station'] == station]
    # normalize columns for plotting
    station_data.drop(columns=['Station', 'Unnamed: 0'], inplace=True)
    station_data.columns = list([col[:-9] for col in station_data.columns])

    # plot station SWE time series
    station_data.T.plot(ax=axis)

    axis.set_title(f'snow fall estimate (SWE) forecasts for station {station}')
    axis.set_xlabel('snow seasons (2020-21)')
    axis.set_ylabel('SWE')

    # output figure as png
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')


@app.route('/plot_date/', methods=['GET', 'POST'])
def plot_date():
    """
    returns png file with plot of interpolated SWE over the encompassing grid of forecasts on given date

    date: URL argument, date for which to show SWE interpolation. if 'random' chosen randomly
    """

    fig = plt.Figure(figsize=(10, 10))
    axis = fig.add_subplot(111)
    # get train points with metadata
    train_points = get_train_metadata("forecasts.csv",
                                      "train_metadata.csv",
                                      'Station')
    # initialize input for interpolation according to date argument from URL
    date = request.args['date']
    if date == 'random':
        date = np.random.choice(train_points.columns[3:])
    elif date not in train_points.columns:
        return 'date not found'
    train_longs = train_points.loc[:, 'longitude']
    train_lats = train_points.loc[:, 'latitude']
    train_swes = train_points.loc[:, date]
    # initialize and fill grid with interpolated data
    grid_longs, grid_lats = generate_grid(train_longs, train_lats)
    grid_data = interpolate_snapshot(train_longs, train_lats, train_swes,
                                     grid_longs, grid_lats)

    # reshape and plot grid data
    grid_data = grid_data.reshape((RES_X, RES_Y))
    im = axis.contourf(grid_data,
                     extent=(grid_longs.min(), grid_longs.max(), grid_lats.min(), grid_lats.max()),
                     levels=20,     cmap='binary')
    # plot forecasts on top
    axis.scatter(train_longs, train_lats,
                 c='blue', marker='o', s=5, alpha=0.6,
                 label='forecasts')

    axis.set_title(f'snow fall estimate (SWE) on {date}')
    axis.set_xlabel('longitude')
    axis.set_ylabel('latitude')
    fig.colorbar(im, fraction=0.0395, pad=0.04)
    axis.legend()

    # output figure as png
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=105)
