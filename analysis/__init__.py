"""
General purpose figures.
"""
from copy import copy

import json
import numpy as np
import os
import pandas as pd
from scipy import stats
import traceback

from data import DataLoader
from db import d_models
from db import connect_and_make_session, empty_tables
from plot import get_n_colors, set_font_size, stacked_overlay

import CONFIG as C
import LOCAL_SETTINGS as L

   
def trace_overlay(
        TRIAL_IDS, VARIABLES, VELOCITY_FILTER,
        REFS=None, COLORS=None, LABELS=None, T_LIM=None, HEIGHT=None):
    """
    Plot overlays of multiple variables for multiple trials on one panel.
    """
    session = connect_and_make_session('mushroom_mushroom')
    trials = session.query(d_models.Trial).filter(
        d_models.Trial.id.in_(TRIAL_IDS)).all()
    dls = get_processed_data_loaders(trials, VELOCITY_FILTER)
    session.close()

    for trial, dl in zip(trials, dls): trial.dl = dl

    ts = [trial.dl.timestamp_gcamp for trial in trials]
    data = [
        {v: getattr(trial.dl, v) for v in VARIABLES}
        for trial in trials
    ]

    if T_LIM is None:
        t_min = np.min([t.min() for t in ts])
        t_max = np.max([t.max() for t in ts])
        T_LIM = [(t_min, t_max)]
    elif not hasattr(T_LIM[0], '__iter__'):
        T_LIM = [T_LIM]

    ax_height = 3*len(trials)
    fig_height = len(T_LIM) * ax_height if HEIGHT is None else HEIGHT

    fig, axs = plt.subplots(
        len(T_LIM), 1, figsize=(15, fig_height), tight_layout=True, squeeze=False)

    for ax, t_lim in zip(axs[:, 0], T_LIM):
        handles = stacked_overlay(
            ax, ts, data, colors=COLORS, labels=LABELS,
            refs=REFS, x_lim=t_lim, lw=2)[0]

    axs[-1, 0].set_xlabel('time (s)')
    axs[0, 0].legend(handles=handles, ncol=2)

    for ax in axs.flatten(): set_fontsize(ax, 16)
    return fig


def trace_overlay_one_trial(
        TRIAL_ID, VARIABLE_SETS, VELOCITY_FILTER,
        REFS=None, COLORS=None, LABELS=None, T_LIM=None,
        HEIGHT=None, SPACING=6, ):
    """
    Plot multiple sets of overlaid variables for a single trial on a single panel.
    """
    session = connect_and_make_session('mushroom_mushroom')
    trial = session.query(d_models.Trial).get(TRIAL_ID)
    trial.dl = get_processed_data_loaders([trial], VELOCITY_FILTER)[0]
    session.close()

    ts = [trial.dl.timestamp_gcamp] * len(VARIABLE_SETS)
    data = [
        {v: getattr(trial.dl, v) for v in variable_set}
        for variable_set in VARIABLE_SETS
    ]

    if T_LIM is None:
        t_min = np.min([t.min() for t in ts])
        t_max = np.max([t.max() for t in ts])
        T_LIM = (t_min, t_max)

    if not hasattr(T_LIM[0], '__iter__'):
        T_LIM = [T_LIM]

    n_rows = len(T_LIM)
    figsize = (15, 1.5*len(VARIABLE_SETS)) if HEIGHT is None else (15, HEIGHT)
    fig, axs = plt.subplots(
        n_rows, 1, figsize=figsize, squeeze=False, tight_layout=True)

    for t_lim, ax in zip(T_LIM, axs[:, 0]):
        all_vars = sum(VARIABLE_SETS, ())
        colors = copy(COLORS)

        for var in all_vars:
            if var not in colors and var in cf.COLORS_DEFAULT:
                colors[var] = cf.COLORS_DEFAULT[var]

        handles = stacked_overlay(
            ax, ts, data, colors=colors, labels=LABELS,
            refs=REFS, x_lim=t_lim, spacing=SPACING, lw=2)[0]

        ax.set_xlabel('time (s)')
        ax.legend(handles=handles, ncol=3)

        set_fontsize(ax, 16)
    return fig
