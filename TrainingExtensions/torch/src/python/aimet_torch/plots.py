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

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


def switch_backend():
    """
    switches to the appropriate backend
    :return: None
    """
    backend = plt.get_backend()
    plt.switch_backend(backend)


def has_display():
    """
    checks to see if there is a display in the current environment
    :return: None
    """
    return "DISPLAY" in os.environ


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


def visualize_module_boxplot(conv_module, name_str, x_stepsize=5, num_y_ticks=20):
    """
    Add boxplots for each output channel on tensorboard under images tab.

    :param conv_module: convNd module
    :param name_str: figure name as will show on tensorboard images tab
    :param: x_stepsize: Size of spacing between values on x axis;the distance between two adjacent values on x axis.
    :param: num_y_ticks: number of evenly spaced samples, calculated over the interval [y_min, y_max].
    :return: None
    """
    if not has_display():
        return
    switch_backend()
    plt.clf()

    arr = get_weights(conv_module)
    fig = plt.figure(figsize=adjust_figure_size(arr.shape[1]))
    plt.boxplot(arr)

    y_min, y_max = np.ndarray.min(arr), np.ndarray.max(arr)
    y_ticks = np.linspace(y_min, y_max, num_y_ticks)
    plt.yticks(y_ticks)
    x_ticks = np.arange(0, arr.shape[1] + 1, x_stepsize)
    plt.xticks(x_ticks, x_ticks)
    plt.xlabel("Output Channels")
    plt.ylabel("Weight Range")
    plt.title("Conv Module Output Channels")
    writer = SummaryWriter("./data")
    writer.add_figure(name_str, fig, walltime=1)
    writer.close()
    return


def visualize_module_lineplot(conv_module, name_str):
    """
    Create a line plot of tensor minimum and maximum lines overlayed on same plot.
    Where x axis is the output channel and the y axis is the min and max values.

    :param conv_module: type convNd module
    :param name_str: figure name as will show on tensorboard images tab
    :return: None
    """

    if not has_display():
        return
    switch_backend()
    plt.clf()
    df_weights = pd.DataFrame(get_weights(conv_module)).describe()

    # fig = plt.figure(figsize=adjust_figure_size(data.shape[1]))
    fig = plt.figure(figsize=(15, 10))
    x = list(range(len(df_weights)))
    plt.plot(x, df_weights["max"])
    plt.plot(x, df_weights["min"])
    # plt.plot(x, df_weights["25th percentile"])
    plt.plot(x, df_weights["50th percentile"])
    # plt.plot(x, df_weights["75th percentile"])

    plt.xlabel("Output Channels")
    plt.ylabel("Weight Range")
    plt.title("Conv Module Output Channels")
    plt.legend(loc='upper right')
    writer = SummaryWriter("./data")
    writer.add_figure(name_str, fig)
    writer.close()


def create_histogram(data, name_str):
    """
    :param data: python list or numpy array value
    :param name_str: figure name as will show on tensorboard images tab
    :return: None
    """

    if not has_display():
        return
    switch_backend()
    plt.clf()

    fig, ax = plt.subplots()
    ax.hist(data)
    write_to_data(name_str, "", fig)
    return


def create_table_from_dataframe(dataframe, name_str):
    """
    Show dataframe on tensorboard under images tab with name as name_str.

    :param dataframe: pandas dataframe to be shown on tensorboard
    :param name_str: figure name as will show on tensorboard images tab
    :return: None
    """
    if not has_display():
        return
    switch_backend()
    fig = plt.figure(figsize=(20, 20))
    fig.patch.set_visible(False)
    plt.axis('off') #[Updated pylint is req. for Python 3.8 positional args spec] pylint: disable=too-many-function-args
    plt.table(cellText=dataframe.values, colLabels=dataframe.columns, loc='top')
    fig.tight_layout()

    write_to_data(name_str, "", fig)


def get_necessary_statistics_from_dataframe(module_weights):
    """
    Generates descriptive statistics summarizing central tendency, dispersion and shape of output channels distribution.

    :param module_weights: module weights represented as a 2d numpy array
    :return: None

    """
    module_weights_as_dataframe = pd.DataFrame(module_weights)
    described_dataframe = module_weights_as_dataframe.describe().drop(index="count")
    return described_dataframe


def compare_overall_model_changes_violinplot(before_module_weights, after_module_weights, tab_name, subplot_name):
    """
    Creates two violin plots, one for all weight ranges before quantization and one for after.

    :param before_module_weights: pandas dataframe of all weights in module before quantization
    :param after_module_weights: pandas dataframe of all weights in module after quantization
    :param tab_name: The name of the tab in which the subplot will show on tensorboard
    :param subplot_name:  The name of the subplot under the tab on tensorboard
    :return: None
    """
    if not has_display():
        return
    switch_backend()
    fig, ax1 = plt.subplots(1, figsize=(8, 5))

    before_weights_flattened = np.ndarray.flatten(before_module_weights).tolist()
    after_weights_flattened = np.ndarray.flatten(after_module_weights).tolist()

    ax1.violinplot([before_weights_flattened, after_weights_flattened], showmeans=True, showextrema=True)

    ax1.set_xticks([1, 2])
    ax1.set_xticklabels(["before quantization ranges", "after quantization ranges"])

    plt.tight_layout()
    write_to_data(tab_name, subplot_name, fig)


def compare_key_stats_scatter_plot(before_module_weights_statistics, after_module_weights_statistics, tab_name,
                                   subplot_name):
    """
    Plots mean vs standard deviation and min vs max befor and after quantization.

    :param before_module_weights_statistics: pandas dataframe of all weights in module before quantization
    :param after_module_weights_statistics: pandas dataframe of all weights in module after quantization
    :param tab_name: The name of the tab in which the subplot will show on tensorboard
    :param subplot_name:  The name of the subplot under the tab on tensorboard
    :return: None
    """
    if not has_display():
        return
    switch_backend()

    plt.clf()

    # Returns a tuple containing a figure and axes object(s).
    # Unpack this tuple into the variables fig and ax1,ax2
    fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(12, 8))

    # Row 1: scatter plots before quantization
    ax1[0].scatter(before_module_weights_statistics.loc['mean'], before_module_weights_statistics.loc['std'], alpha=.3,
                   color='orange')
    ax1[0].set(xlabel='mean weights', ylabel='std weights', title='Before Quantization Mean vs Std')
    ax1[1].scatter(before_module_weights_statistics.loc['min'], before_module_weights_statistics.loc['max'], alpha=.3,
                   color='steelblue')
    ax1[1].set(xlabel='min weights', ylabel='max weights', title='Before Quantization Min vs Max')

    # Row 2: scatter plots after quantization
    ax2[0].scatter(after_module_weights_statistics.loc['mean'], after_module_weights_statistics.loc['std'], alpha=.3,
                   color='orange')
    ax2[0].set(xlabel='mean weights', ylabel='std weights', title='After Quantization Mean vs Std')
    ax2[1].scatter(after_module_weights_statistics.loc['min'], after_module_weights_statistics.loc['max'], alpha=.3,
                   color='steelblue')
    ax2[1].set(xlabel='min weights', ylabel='max weights', title='After Quantization Min vs Max')

    y_lower_bound = min(np.min(before_module_weights_statistics.loc['max']),
                        np.min(after_module_weights_statistics.loc['max']))
    y_upper_bound = max(np.max(before_module_weights_statistics.loc['max']),
                        np.max(after_module_weights_statistics.loc['max']))

    x_lower_bound = min(np.min(before_module_weights_statistics.loc['min']),
                        np.min(after_module_weights_statistics.loc['min']))
    x_upper_bound = max(np.max(before_module_weights_statistics.loc['min']),
                        np.max(after_module_weights_statistics.loc['min']))

    ax1[1].set_ylim(y_lower_bound - .1, y_upper_bound + .1)
    ax1[1].set_xlim(x_lower_bound - .1, x_upper_bound + .1)

    ax2[1].set_ylim(y_lower_bound - .1, y_upper_bound + .1)
    ax2[1].set_xlim(x_lower_bound - .1, x_upper_bound + .1)

    plt.tight_layout()
    write_to_data(tab_name, subplot_name, fig)
    return


def compare_overall_changes_line_plot(before_module_weights_statistics, after_module_weights_statistics, tab_name,
                                      subplot_name):
    """
    Compares the weight ranges before and after quantization of conv2d and linear modules,
    given pandas dataframes before and after quantization

    :param before_module_weights_statistics: pandas dataframe of all weights in module before quantization
    :param after_module_weights_statistics: pandas dataframe of all weights in module after quantization
    :param tab_name: The name of the tab in which the subplot will show on tensorboard
    :param subplot_name:  The name of the subplot under the tab on tensorboard
    :return: None
    """
    if not has_display():
        return
    switch_backend()
    fig, ax1 = plt.subplots(1, figsize=(14, 12))

    count_col = before_module_weights_statistics.shape[1]  # count number of columns

    output_channels = list(range(count_col))
    ax1.plot(output_channels, before_module_weights_statistics.loc['min'], color='khaki', label="min")
    ax1.plot(output_channels, after_module_weights_statistics.loc['min'], color='darkgoldenrod', label="new minimum")
    ax1.fill_between(output_channels, before_module_weights_statistics.loc['min'], after_module_weights_statistics.loc['min'],
                     color='orange', alpha=0.2)

    ax1.plot(output_channels, before_module_weights_statistics.loc['mean'], color='steelblue', label="mean")
    ax1.plot(output_channels, after_module_weights_statistics.loc['mean'], color='darkcyan', label="new mean")
    ax1.fill_between(output_channels, before_module_weights_statistics.loc['mean'], after_module_weights_statistics.loc['mean'],
                     color='steelblue', alpha=0.2)

    ax1.plot(output_channels, before_module_weights_statistics.loc['max'], color='lightgreen', label="max")
    ax1.plot(output_channels, after_module_weights_statistics.loc['max'], color='green', label="new max")
    ax1.fill_between(output_channels, before_module_weights_statistics.loc['max'], after_module_weights_statistics.loc['max'],
                     color='green', alpha=0.2)

    ax1.legend(loc="upper right")

    plt.tight_layout()
    write_to_data(tab_name, subplot_name, fig)


def write_to_data(tab_name, subplot_name, fig):
    """
    Writes the figure object as an event file in the data directory.

    :param tab_name: Name of tab on tensorboard
    :param subplot_name: Name of subplot inside tab
    :param fig: Figure object on matplotlib
    :return: None
    """
    tag = tab_name + "/" + subplot_name
    if subplot_name == "":
        tag = tab_name
    writer = SummaryWriter("./data")
    writer.add_figure(tag, fig)
    writer.close()


def compare_boxplots_before_after_quantization(before_data, after_data, tab_name, subplot_name):
    """
    Compares the weight ranges before and after quantization of conv2d and linear modules, \
    given a 2d numpy array of weights before and after quantization.

    :param before_data: Before quantization weights numpy array
    :param after_data: After quantization weights  numpy array
    :param tab_name: The name of the tab in which the subplot will show on tensorboard
    :param subplot_name:  The name of the subplot under the tab on tensorboard
    :return: None
    """
    if not has_display():
        return
    switch_backend()

    count_col = before_data.shape[1]
    fig, (ax1, ax2) = plt.subplots(2, figsize=adjust_figure_size(count_col))

    ax1.boxplot(before_data, patch_artist=True)
    ax2.boxplot(after_data, patch_artist=True)

    xticks = np.arange(0, count_col + 1, 10)
    ax1.xaxis.set_ticks(xticks)
    ax1.set_xticklabels(xticks)

    ax2.xaxis.set_ticks(xticks)
    ax2.set_xticklabels(xticks)

    ax1.set(xlabel="Output Channels Before Quantization", ylabel="Weight Ranges")
    ax2.set(xlabel="Output Channels After Quantization", ylabel="Weight Ranges")
    plt.tight_layout()

    write_to_data(tab_name, subplot_name, fig)


def map_all_module_weights(model):
    """
    Returns a python dictionary mapping each conv2d module and linear module in the input model to its weights.

    :param model: pytorch model
    :return: None
    """
    module_weights_map = {}
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.modules.conv.Conv2d, torch.nn.modules.linear.Linear)):
            module_weights = get_weights(module)
            module_weights_map[name] = module_weights
    return module_weights_map


def adjust_figure_size(num_channels):
    """
    Adjusts the figure size of single plot graphs
    :param num_channels: Number of channels to be plotted
    :return: A tuple, to be passed into figure size so that all channels can be displayed in a plot.
    """
    base_length = 15
    base_width = 10
    num_inches_to_add = num_channels / 60

    length = base_length + num_inches_to_add
    return length, base_width


def before_after_plots_for_quantized_model(before_weights_map, after_weights_map):
    """
    Creates two event files, one for boxplots and another for all other visualization for quantization.
    :param before_weights_map: python dictionary where module name is key and values are weights before quantization
    :param after_weights_map:  python dictionary where module name is key and values are weights before quantization
    :return: None
    """

    for key in before_weights_map.keys():
        before_quantization_data = before_weights_map[key]
        after_quantization_data = after_weights_map[key]
        compare_boxplots_before_after_quantization(before_quantization_data, after_quantization_data,
                                                   tab_name=key, subplot_name="Boxplots")

        before_quantization_as_dataframe = get_necessary_statistics_from_dataframe(before_quantization_data)
        after_quantization_as_dataframe = get_necessary_statistics_from_dataframe(after_quantization_data)

        compare_overall_model_changes_violinplot(before_quantization_data, after_quantization_data, tab_name=key,
                                                 subplot_name="Violin")
        compare_overall_changes_line_plot(before_quantization_as_dataframe, after_quantization_as_dataframe,
                                          tab_name=key,
                                          subplot_name="Line")
        compare_key_stats_scatter_plot(before_quantization_as_dataframe, after_quantization_as_dataframe, tab_name=key,
                                       subplot_name="Scatter")


def clear_event_files(my_path):
    """
    Removes all tensorflow event files in the specified directory
    :param my_path: path from current directory
    :return: None
    """

    for root_dir_file_tuple in os.walk(my_path):
        root = root_dir_file_tuple[0]
        files = root_dir_file_tuple[2]
        for file in files:
            if file[:19] == "events.out.tfevents":
                os.remove(os.path.join(root, file))
