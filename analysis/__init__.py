"""
General purpose figures.
"""
import matplotlib.pyplot as plt; plt.style.use('ggplot')
from IPython.display import display, HTML
from copy import copy

import json
import numpy as np
import os
import pandas as pd
from scipy import stats
import traceback

from db import d_models
from db import connect_and_make_session, empty_tables
from plot import get_n_colors, set_font_size, stacked_overlay
from shortcuts import get_processed_data_loaders

import CONFIG as C
import LOCAL_SETTINGS as L


def write_trials_from_file_system():
    """
    Look in the file system and extract paths, etc., of all the data files and store them
    in the database.
    """

    session = connect_and_make_session()
    empty_tables(session, d_models.Trial)

    # get directories for different experiments

    for expt_dir in next(os.walk(L.DATA_ROOT))[1]:
        print('LOADING DATA FILES FROM EXPERIMENT: "{}".'.format(expt_dir))

        # get all of the bottom level directories (which contain the data files)
        data_dirs = []

        for root, dirs, files in os.walk(os.path.join(L.DATA_ROOT, expt_dir)):
            if not dirs: data_dirs.append(root)

        # store trial for each data directory in database
        data_ctr = 0

        for data_dir in data_dirs:

            print('TRIAL: "{}"'.format(os.path.basename(data_dir)))

            trial = d_models.Trial()

            trial.name = os.path.basename(data_dir)
            trial.expt = expt_dir
            trial.fly = '.'.join(trial.name.split('.')[:2])
            trial.number = int(trial.name.split('.')[2])

            trial.path = os.path.relpath(data_dir, L.DATA_ROOT)

            # get names of all files in trial directory
            file_names = next(os.walk(data_dir))[2]

            try:
                file_name_behav = [
                    fn for fn in file_names if fn.endswith(C.BEHAV_FILE_ENDING)][0]
            except:
                raise Exception(
                    'Could not find ".dat" file in directory: "{}"'.format(
                        data_dir))

            if True:
                try:
                    file_name_gcamp = [
                        fn for fn in file_names if fn == C.ROI_PROFILES_FILE][0]
                except:
                    raise Exception(
                        ('Could not find "ROI-profiles" file in directory: '
                         '"{}"'.format(data_dir)))

                try:
                    file_name_gcamp_timestamp = [
                        fn for fn in file_names if fn == C.GCAMP_FILE
                    ][0]
                except:
                    raise Exception(
                        ('Could not find GCaMP timestamp file in directory: '
                         '"{}"'.format(data_dir)))

                try:
                    file_name_light_times = [
                        fn for fn in file_names if fn == C.LIGHT_TIMES_FILE
                    ][0]
                except:
                    raise Exception(
                        ('Could not find light times file in directory: '
                         '"{}"'.format(data_dir)))

            else:
                try:
                    file_name_gcamp = [
                        fn for fn in file_names
                        if fn.endswith('.csv') and 'Auto' not in fn][0]
                except:
                    raise Exception(
                        'Could not find ".csv" file in directory: "{}"'.format(
                            data_dir))

                file_name_gcamp_timestamp = None
                file_name_light_times = None

            # see if there are files defining bouts and turns
            file_name_bouts = \
                'Auto_Bouts.csv' if 'Auto_Bouts.csv' in file_names else None
            file_name_left_turns = \
                'Auto_TurnL.csv' if 'Auto_TurnL.csv' in file_names else None
            file_name_right_turns = \
                'Auto_TurnR.csv' if 'Auto_TurnR.csv' in file_names else None

            # see if there is a file defining air tube motion
            file_name_air_tube = C.AIR_TUBE_FILE \
                if C.AIR_TUBE_FILE in file_names else None

            # see if there is a details file
            if 'details.json' in file_names:
                path = os.path.join(data_dir, 'details.json')
                with open(path) as f: details = json.load(f)
            else:
                details = {}

            trial.file_name_behav = file_name_behav
            trial.file_name_gcamp = file_name_gcamp
            trial.file_name_gcamp_timestamp = file_name_gcamp_timestamp
            trial.file_name_light_times = file_name_light_times

            trial.file_name_bouts = file_name_bouts
            trial.file_name_left_turns = file_name_left_turns
            trial.file_name_right_turns = file_name_right_turns

            trial.file_name_air_tube = file_name_air_tube
            trial.details = details

            session.add(trial)

            data_ctr += 1

        session.commit()

        print(('{}/{} TRIALS LOADED FOR EXPERIMENT: '
               '"{}".'.format(data_ctr, len(data_dirs), expt_dir)))

    print('Complete.')


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
