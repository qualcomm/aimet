# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2019, Qualcomm Innovation Center, Inc. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

""" Create visualizations on the weights in each conv and linear layer in a model"""
import math
import holoviews as hv
import numpy as np
import pandas as pd
import torch
from bokeh.layouts import row
from bokeh.models import HoverTool, ColumnDataSource, Span, TableColumn, DataTable
from bokeh.plotting import figure
from bokeh.layouts import column
from bokeh.models import Div

# Some magic stuff happening during import that ties pandas dataframe to hvplot
# Need this import, please don't remove
import hvplot.pandas  # pylint:disable=unused-import

from aimet_common.plotting_utils import style
from aimet_common import bokeh_plots
from aimet_torch.utils import get_device


def get_weights(conv_module):
    """
    Returns the weights of a conv_module in a 2d matrix, where each column is an output channel.

    :param conv_module: convNd module
    :return: 2d numpy array
    """
    axis_0_length = conv_module.weight.shape[0]
    axis_1_length = np.prod(conv_module.weight.shape[1:])
    reshaped = conv_module.weight.reshape(int(axis_0_length), int(axis_1_length))
    weights = reshaped.detach().numpy().T
    return weights


def map_all_module_weights_to_data_frame(model):
    """
    Returns a python dictionary mapping each conv and linear module in the input model to a data frame of its weights.

    :param model: pytorch model
    :return: python dictionary
    """
    module_weights_map = {}
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.modules.conv.Conv2d, torch.nn.modules.linear.Linear)):
            module_weights = get_weights(module)
            module_weights_data_frame = pd.DataFrame(module_weights)
            module_weights_map[name] = module_weights_data_frame
    return module_weights_map


def line_plot_changes_in_summary_stats(data_before, data_after, x_axis_label=None, y_axis_label=None, title=None):
    """
    Returns a bokeh figure object showing a lineplot of min, max, and mean per output channel, shading in the area
    difference between before and after.
    :param data_before: pandas data frame with columns min, max, and mean.
    :param data_after: pandas data frame with columns min, max, and mean
    :param x_axis_label: string description of x axis
    :param y_axis_label: string description of y axis
    :param title: title for the plot
    :return: bokeh figure object
    """
    layer_weights_old_model = convert_pandas_data_frame_to_bokeh_column_data_source(data_before)
    layer_weights_new_model = convert_pandas_data_frame_to_bokeh_column_data_source(data_after)

    plot = figure(x_axis_label=x_axis_label, y_axis_label=y_axis_label,
                  title=title,
                  tools="pan, box_zoom, crosshair, reset, save",
                  width=950, height=600, sizing_mode='stretch_both', output_backend="webgl")
    plot.line(x='index', y='min', line_width=2, line_color="#2171b5", line_dash='dotted', legend_label="Before Optimization",
              source=layer_weights_old_model, name="old model")
    plot.line(x='index', y='max', line_width=2, line_color="green", line_dash='dotted', source=layer_weights_old_model,
              name="old model")
    plot.line(x='index', y='mean', line_width=2, line_color="orange", line_dash='dotted',
              source=layer_weights_old_model, name="old model")

    plot.line(x='index', y='min', line_width=2, line_color="#2171b5",
              legend_label="After Optimization", source=layer_weights_new_model, name="new model")
    plot.line(x='index', y='max', line_width=2, line_color="green",
              source=layer_weights_new_model, name="new model")
    plot.line(x='index', y='mean', line_width=2, line_color="orange",
              source=layer_weights_new_model, name="new model")

    plot.varea(x=data_after.index,
               y1=data_after['min'],
               y2=data_before['min'], fill_alpha=0.3, legend_label="shaded region", name="new model")

    plot.varea(x=data_after.index,
               y1=data_after['max'],
               y2=data_before['max'], fill_color="green", fill_alpha=0.3, legend_label="shaded region")

    plot.varea(x=data_after.index,
               y1=data_after['mean'],
               y2=data_before['mean'], fill_color="orange", fill_alpha=0.3, legend_label="shaded region")

    plot.legend.location = "top_left"
    plot.legend.click_policy = "hide"
    plot.legend.background_fill_alpha = 0.3

    if not x_axis_label or not y_axis_label or not title:
        layout = row(plot)
        return layout

    # display a tooltip whenever the cursor in line with a glyph
    hover1 = HoverTool(tooltips=[("Output Channel", "$index"),
                                 ("Mean Before Optimization", "@mean{0.00}"),
                                 ("Minimum Before Optimization", "@min{0.00}"),
                                 ("Maximum Before Optimization", "@max{0.00}"),
                                 ("25 Percentile Before Optimization", "@{25%}{0.00}"),
                                 ("75 Percentile Before Optimization", "@{75%}{0.00}")], name='old model',
                       mode='mouse'
                       )
    hover2 = HoverTool(tooltips=[("Output Channel", "$index"),
                                 ("Mean After Optimization", "@mean{0.00}"),
                                 ("Minimum After Optimization", "@min{0.00}"),
                                 ("Maximum After Optimization", "@max{0.00}"),
                                 ("25 Percentile After Optimization", "@{25%}{0.00}"),
                                 ("75 Percentile After Optimization", "@{75%}{0.00}")], name='new model',
                       mode='mouse'
                       )
    plot.add_tools(hover1)
    plot.add_tools(hover2)
    style(plot)

    layout = row(plot)
    return layout


def line_plot(x, y, x_axis_label, y_axis_label, title, x_range=None):
    """
    :param x: x coordinates of data points
    :param y: y coordinates of data points
    :param x_axis_label: string description of x axis
    :param y_axis_label: string description of y axis
    :param title: title for the plot
    :return: bokeh figure object
    """
    plot = figure(x_axis_label=x_axis_label, y_axis_label=y_axis_label,
                  title=title,
                  tools="pan, box_zoom, crosshair, reset, save",
                  x_range=x_range,
                  width=1500)
    plot.line(x=x, y=y, line_width=2, line_color="#2171b5")
    plot.circle(x=x, y=y, color="black", alpha=0.7, size=10)
    if isinstance(x_range, list) and isinstance(x_range[0], str):
        plot.xaxis.major_label_orientation = math.pi / 4

    style(plot)

    return plot


def scatter_plot_summary_stats(data_frame, x_axis_label_mean="mean", y_axis_label_mean="standard deviation",
                               title_mean="Mean vs Standard Deviation",
                               x_axis_label_min="Minimum",
                               y_axis_label_min="Maximum", title_min="Minimum vs Maximum"):
    """
    Creates a scatter plot, plotting min vs max, and mean vs std side by side.
    :param data_frame: pandas data frame object
    :param x_axis_label_mean: string description of x axis in plot showing mean vs std
    :param y_axis_label_mean: string description of y axis in plot showing mean vs std
    :param x_axis_label_min: string description of x axis in plot showing min vs max
    :param y_axis_label_min: string description of y axis in plot showing min vs max
    :return: bokeh figure
    """
    plot1 = figure(x_axis_label=x_axis_label_mean, y_axis_label=y_axis_label_mean,
                   title=title_mean,
                   tools="box_zoom, crosshair,reset", output_backend="webgl")
    plot1.circle(x=data_frame['mean'], y=data_frame['std'], size=10, color="orange", alpha=0.4)

    plot2 = figure(x_axis_label=x_axis_label_min, y_axis_label=y_axis_label_min,
                   title=title_min,
                   tools="box_zoom, crosshair,reset", output_backend="webgl")
    plot2.circle(x=data_frame['min'], y=data_frame['max'], size=10, color="#2171b5", alpha=0.4)
    style(plot1)
    style(plot2)
    # layout = row(plot1, plot2)
    return plot1, plot2


def box_plot_max_ranges(data_frame, output_channels_needed, x_label=None, y_label=None, title=None):
    """
    Creates a figure with n boxplots that can be most sensitive to outliers.
    :param data_frame: pandas dataframe object
    :param described_df: pandas dataframe object with a max and min column
    :param largest_ranges_n: number of boxplots to be made
    :return: a boxplot figure that has n boxplots with the largest range
    """
    # CLEAN THIS UP
    data_frame.columns = data_frame.columns.map(str)
    max_range_df = data_frame[output_channels_needed]
    columns = list(max_range_df.columns)
    plot = max_range_df.hvplot.box(y=columns, legend=False, invert=False, box_fill_alpha=0.5,
                                   outlier_fill_color="red",
                                   outlier_alpha=0.3, width=1200, height=600,
                                   xlabel=x_label,
                                   ylabel=y_label,
                                   title=title)
    bokeh_plot = hv.render(plot)

    style(bokeh_plot)
    return bokeh_plot


def identify_max_range_columns(data_frame, described_df, num_columns=50):
    """
    Returns a list of columns with the maximum absolute ranges.
    :param data_frame: pandas data frame
    :param described_df: pandas data frame with summary statistics
    :param num_columns: number of output channels to return
    :return: list of output channels with maximum ranges.
    """
    data_frame.columns = data_frame.columns.map(str)
    described_df['range'] = described_df['max'] - described_df['min']
    described_df = described_df.sort_values(by=['range'], ascending=False)
    output_channels_needed = described_df[:num_columns].index

    output_channels_needed = [str(i) for i in output_channels_needed]
    return output_channels_needed


def visualize_ranges_before_after_for_model(before_weights_map, after_weights_map):
    """
    Adds plots to display and compare weight ranges for a model, before and after quantization.

    This function was used for demo-ing but creating the boxplots takes way too long so it was removed from being a
    high level function.

    :param before_weights_map: python dictionary where module name is key and values are a pandas data frame containing
    weights before quantization
    :param after_weights_map:  python dictionary where module name is key and values are a pandas data frame containing
    weights after quantization
    :return:
    """

    for key in before_weights_map.keys():
        before_quantization_data_weights = before_weights_map[key]
        after_quantization_data_weights = after_weights_map[key]

        before_quantization_data = before_quantization_data_weights.describe().T
        after_quantization_data = after_quantization_data_weights.describe().T

        max_range_output_channels = identify_max_range_columns(before_quantization_data_weights,
                                                               before_quantization_data)

        layout = bokeh_plots.PlotsLayout()
        layout.title = key
        layout.add_row(scatter_plot_summary_stats(before_quantization_data,
                                                  x_axis_label_mean="Mean Weights Per Output Channel",
                                                  y_axis_label_mean="Std Per Output Channel",
                                                  title_mean="Mean vs Std After CLE",
                                                  x_axis_label_min="Min Weights Per Output Channel",
                                                  y_axis_label_min="Max Weights Per Output Channel",
                                                  title_min="Min vs Max After CLE"))
        layout.add_row(scatter_plot_summary_stats(after_quantization_data,
                                                  x_axis_label_mean="Mean Weights Per Output Channel",
                                                  y_axis_label_mean="Std Per Output Channel",
                                                  title_mean="Mean vs Std Before CLE",
                                                  x_axis_label_min="Min Weights Per Output Channel",
                                                  y_axis_label_min="Max Weights Per Output Channel",
                                                  title_min="Min vs Max Before CLE"))
        layout.add_row(line_plot_changes_in_summary_stats(before_quantization_data, after_quantization_data,
                                                          x_axis_label="Output Channel",
                                                          y_axis_label="Summary statistics",
                                                          title="Changes in Key Stats Per Output Channel"))
        layout.add_row(box_plot_max_ranges(before_quantization_data_weights, max_range_output_channels,
                                           x_label="Output Channels",
                                           y_label="Weight Ranges",
                                           title="Before CLE: Output Channels With Largest Total Range"))
        layout.add_row(box_plot_max_ranges(after_quantization_data_weights, max_range_output_channels,
                                           x_label="Output Channels",
                                           y_label="Weight Ranges",
                                           title="After CLE: Output Channels With Largest Total Range"))
        return layout.complete_layout()


def line_plot_summary_statistics_model(layer_name, layer_weights_data_frame, height, width):
    """
    Given a layer
    :param layer_name:
    :param layer_weights_data_frame:
    :return:
    """
    layer_weights = convert_pandas_data_frame_to_bokeh_column_data_source(layer_weights_data_frame)
    plot = figure(x_axis_label="Output Channels", y_axis_label="Summary Statistics",
                  title="Weight Ranges per Output Channel: " + layer_name,
                  tools="pan, box_zoom, crosshair, reset, save",
                  width=width, height=height, output_backend="webgl")
    plot.line(x='index', y='min', line_width=2, line_color="#2171b5",
              legend_label="Minimum", source=layer_weights)
    plot.line(x='index', y='max', line_width=2, line_color="green",
              legend_label="Maximum", source=layer_weights)
    plot.line(x='index', y='mean', line_width=2, line_color="orange",
              legend_label="Average", source=layer_weights)

    plot.legend.location = "top_left"
    plot.legend.click_policy = "hide"
    plot.legend.background_fill_alpha = 0.3

    plot.add_tools(HoverTool(tooltips=[("Output Channel", "$index"),
                                       ("Mean", "@mean{0.00}"),
                                       ("Min", "@min{0.00}"),
                                       ("Max", "@max{0.00}"),
                                       ("25 percentile", "@{25%}{0.00}"),
                                       ("75 percentile", "@{75%}{0.00}")],
                             # display a tooltip whenever the cursor is vertically in line with a glyph
                             mode='mouse'
                             ))
    style(plot)
    return plot


def identify_problematic_output_channels(module_weights_data_frame_described):
    """
    return a list of output channels that have large weight ranges
    :param module_weights_data_frame: pandas data frame where each column are summary statistics for each row, output channels
    :param largest_ranges_n: number of output channels to return
    :return:
    """
    # data_frame.columns = data_frame.columns.map(str)
    module_weights_data_frame_described['range'] = module_weights_data_frame_described['max'] - \
                                                   module_weights_data_frame_described['min']
    module_weights_data_frame_described["abs range"] = module_weights_data_frame_described["range"].abs()
    variable = module_weights_data_frame_described["abs range"].min()
    module_weights_data_frame_described["relative range"] = module_weights_data_frame_described["abs range"] / variable
    described_df = module_weights_data_frame_described.sort_values(by=['relative range'], ascending=False)
    all_output_channel_ranges = described_df["relative range"]
    output_channels_needed = detect_outlier_channels(all_output_channel_ranges)

    return output_channels_needed, all_output_channel_ranges


def detect_outlier_channels(data_frame_with_relative_ranges):
    """
    Detects outliers for relative weight ranges.
    :param data_frame_with_relative_ranges: pandas data frame with column name "relative ranges"
    :return: list of output channels that have very large weight ranges
    """
    Q1 = data_frame_with_relative_ranges.quantile(0.25)
    Q3 = data_frame_with_relative_ranges.quantile(0.75)
    IQR = Q3 - Q1
    v = data_frame_with_relative_ranges > (Q3 + 1.5 * IQR)
    v_df = v.to_frame()
    keep_only_outliers = v_df.loc[v_df['relative range']]
    output_channels_list = keep_only_outliers.index
    return output_channels_list


def add_vertical_line_to_figure(x_coordinate, figure_object):
    """
    adds a vertical line to a bokeh figure object
    :param x_coordinate: x_coordinate to add line
    :param figure_object: bokeh figure object
    :return: None
    """
    # Vertical line
    vertical_line = Span(location=x_coordinate, dimension='height', line_color='red', line_width=1)
    figure_object.add_layout(vertical_line)


def histogram(data_frame, column_name, num_bins, x_label=None, y_label=None, title=None):
    """
    Creates a histogram of the column in the input data frame.
    :param data_frame: pandas data frame
    :param column_name: column in data frame
    :param num_bins: number of bins to divide data into for histogram
    :return: bokeh figure object
    """
    hv_plot_object = data_frame.hvplot.hist(column_name, bins=num_bins, height=400, tools="", xlabel=x_label,
                                            ylabel=y_label,
                                            title=title, fill_alpha=0.5)

    bokeh_plot = hv.render(hv_plot_object)
    style(bokeh_plot)
    return bokeh_plot


def convert_pandas_data_frame_to_bokeh_data_table(data):
    """
    Converts a pandas data frame to a bokeh column data source object so that it can be pushed to a server document
    :param data: pandas data frame
    :return: data table that can be displayed on a bokeh server document
    """
    data["index"] = data.index
    data = data[['index'] + data.columns[:-1].tolist()]

    data.columns.map(str)
    source = ColumnDataSource(data=data)
    columns = [TableColumn(field=column_str, title=column_str) for column_str in data.columns]  # bokeh columns
    data_table = DataTable(source=source, columns=columns)
    layout = add_title(data_table, "Table Summarizing Weight Ranges")
    return layout


def convert_pandas_data_frame_to_bokeh_column_data_source(data):
    """
    Converts a pandas data frame to a bokeh column data source object so that it can be pushed to a server document
    :param data: pandas data frame
    :return: data table that can be displayed on a bokeh server document
    """
    data["index"] = data.index
    data = data[['index'] + data.columns[:-1].tolist()]

    data.columns.map(str)
    source = ColumnDataSource(data=data)
    return source


def add_title(layout, title):
    """
    Add a title to the layout.
    :return: layout wrapped with title div.
    """
    text_str = "<b>" + title + "</b>"
    wrap_layout_with_div = column(Div(text=text_str), layout)
    return wrap_layout_with_div


def visualize_weight_ranges_single_layer(layer, layer_name, scatter_plot=False):
    """
    Given a layer, visualizes weight ranges with scatter plots and line plots
    :param layer: layer with weights
    :param layer_name: layer name
    :param scatter_plot: Include scatter plot in plots
    :return: None
    """
    device = get_device(layer)
    layer.cpu()
    layer_weights = pd.DataFrame(get_weights(layer))
    layer_weights_summary_statistics = layer_weights.describe().T

    line_plots = line_plot_summary_statistics_model(layer_name=layer_name,
                                                    layer_weights_data_frame=layer_weights_summary_statistics,
                                                    width=1000, height=700)

    if scatter_plot:

        scatter_plot_mean, scatter_plot_min = scatter_plot_summary_stats(layer_weights_summary_statistics,
                                                                         x_axis_label_mean="Mean Weights Per Output Channel",
                                                                         y_axis_label_mean="Std Per Output Channel",
                                                                         title_mean="Mean vs Standard Deviation: " + layer_name,
                                                                         x_axis_label_min="Min Weights Per Output Channel",
                                                                         y_axis_label_min="Max Weights Per Output Channel",
                                                                         title_min="Minimum vs Maximum: " + layer_name)

        scatter_plots_layout = row(scatter_plot_mean, scatter_plot_min)

        layout = column(scatter_plots_layout, line_plots)
    else:
        layout = line_plots
    layout_with_title = add_title(layout, layer_name)

    # Move layer back to device
    layer.to(device=device)
    return layout_with_title


def visualize_relative_weight_ranges_single_layer(layer, layer_name):
    """
    publishes a line plot showing  weight ranges for each layer, summary statistics
    for relative weight ranges, and a histogram showing weight ranges of output channels

    :param model: p
    :return:
    """
    layer_weights_data_frame = pd.DataFrame(get_weights(layer)).describe().T
    plot = line_plot_summary_statistics_model(layer_name, layer_weights_data_frame, width=1150, height=700)

    # list of problematic output channels, data frame containing magnitude of range in each output channel
    problematic_output_channels, output_channel_ranges_data_frame = identify_problematic_output_channels(
        layer_weights_data_frame)

    histogram_plot = histogram(output_channel_ranges_data_frame, "relative range", 75,
                               x_label="Weight Range Relative to Smallest Output Channel",
                               y_label="Count",
                               title="Relative Ranges For All Output Channels")
    output_channel_ranges_data_frame = output_channel_ranges_data_frame.describe().T.to_frame()
    output_channel_ranges_data_frame = output_channel_ranges_data_frame.drop("count")

    output_channel_ranges_as_column_data_source = convert_pandas_data_frame_to_bokeh_data_table(
        output_channel_ranges_data_frame)

    # add vertical lines to highlight problematic channels
    for channel in problematic_output_channels:
        add_vertical_line_to_figure(channel, plot)

    # push plot to server document
    column_layout = column(histogram_plot, output_channel_ranges_as_column_data_source)
    layout = row(plot, column_layout)
    layout_with_title = add_title(layout, layer_name)

    return layout_with_title


def visualize_changes_after_optimization_single_layer(name, old_model_module, new_model_module, scatter_plot=False):
    """
    Creates before and after plots for a given layer.
    :param name: name of module
    :param old_model_module: the module of the model before optimization
    :param new_model_module: the module of the model after optimization
    :param scatter_plot: Include scatter plot in plots
    :return: None
    """

    device_old_module = get_device(old_model_module)
    device_new_module = get_device(new_model_module)

    old_model_module.cpu()
    new_model_module.cpu()

    layout = bokeh_plots.PlotsLayout()
    layout.title = name
    layer_weights_summary_statistics_old = pd.DataFrame(get_weights(old_model_module)).describe().T
    layer_weights_summary_statistics_new = pd.DataFrame(get_weights(new_model_module)).describe().T

    summary_stats_line_plot = line_plot_changes_in_summary_stats(layer_weights_summary_statistics_old,
                                                                 layer_weights_summary_statistics_new,
                                                                 x_axis_label="Output Channel",
                                                                 y_axis_label="Summary statistics",
                                                                 title="Changes in Key Stats Per Output Channel")

    if scatter_plot:
        plot_mean_old_model, plot_min_old_model = scatter_plot_summary_stats(layer_weights_summary_statistics_old,
                                                                             x_axis_label_mean="Mean Weights Per Output Channel",
                                                                             y_axis_label_mean="Std Per Output Channel",
                                                                             title_mean="Mean vs Std After Optimization",
                                                                             x_axis_label_min="Min Weights Per Output Channel",
                                                                             y_axis_label_min="Max Weights Per Output Channel",
                                                                             title_min="Min vs Max After Optimization")

        plot_mean_new_model, plot_min_new_model = scatter_plot_summary_stats(layer_weights_summary_statistics_new,
                                                                             x_axis_label_mean="Mean Weights Per Output Channel",
                                                                             y_axis_label_mean="Std Per Output Channel",
                                                                             title_mean="Mean vs Std Before Optimization",
                                                                             x_axis_label_min="Min Weights Per Output Channel",
                                                                             y_axis_label_min="Max Weights Per Output Channel",
                                                                             title_min="Min vs Max Before Optimization")
        layout.add_row(row(plot_mean_old_model, plot_mean_new_model, plot_min_old_model))

        layout.add_row(row(summary_stats_line_plot, plot_min_new_model))
    else:
        layout.add_row(summary_stats_line_plot)

    old_model_module.to(device=device_old_module)
    new_model_module.to(device=device_new_module)

    return layout.complete_layout()
