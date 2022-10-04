"""
    plotting some of the results ...
"""
import logging

import numpy as np
import pandas as pd
import seaborn as sns

import plot_utility
plot_utility.init_plot()
from matplotlib import pyplot as plt


def tensorarray_to_array(tensor_array):
    if hasattr(tensor_array[0], 'data'):
        return [elem.data.tolist() for elem in tensor_array]
    else:
        return tensor_array


def plot_sim_values(gtp_sim_val, gtn_sim_val, plot_filebase, size=1, median=None):
    """
    plot similarity values for same vs. different author utterances

    :param gtp_sim_val: array with ground truth positive (i.e., same author) similarity values
    :param gtn_sim_val: array with ground truth negative (i.e., distinct author) similarity values
    :param plot_filebase: folder to where plot should be saved
    :param size: size of plot points
    :return:
    """
    plt.figure()
    # Prepare data as pandas dataframe
    df = pd.DataFrame({'Label': [0]*(len(gtp_sim_val)) + [1]*(len(gtn_sim_val)),
                        'Similarity Value': tensorarray_to_array(gtp_sim_val) + tensorarray_to_array(gtn_sim_val)})

    sns.set_theme(style="ticks")
    graph = sns.catplot(x='Label', y='Similarity Value', data=df, s=size)  # dropna=True)
    x_label = "Same Author"
    y_label = "Similarity Values"
    graph.set_axis_labels(x_label, y_label)
    plt.tight_layout()

    # adding a median line if given
    if median:
        xmin, xmax = plt.xlim()
        plt.hlines(median, xmin=xmin, xmax=xmax, colors="green")


    logging.info("Saving Figure to " + str(plot_filebase) + ".png")
    plt.savefig(plot_filebase + ".png")
    df.to_pickle(plot_filebase + ".pickle")
    plt.close('all')

    plt.figure()

    # Prepare data as pandas dataframe
    df = pd.DataFrame({'Distinct Author': [0]*(len(gtp_sim_val)) + [1]*(len(gtn_sim_val)),
                        'Similarity Value': tensorarray_to_array(gtp_sim_val) + tensorarray_to_array(gtn_sim_val)})
    graph = sns.displot(df, x="Similarity Value", hue="Distinct Author", kind="kde", fill=True)


    x_label = "Similarity Value"
    y_label = "Density"
    graph.set_axis_labels(x_label, y_label)
    plt.tight_layout()

    logging.info("Saving Figure to " + str(plot_filebase) + "2.png")
    plt.savefig(plot_filebase + "2.png")
    df.to_pickle(plot_filebase + "2.pickle")
    plt.close('all')
