"""Some constants for this project."""
PI = 3.14159265359
DT = 0.1

EXPTS = [
    'asensory_5_hz',
    'asensory_10_hz',
    'closed',
    'sinusoidal',
    'closed_white',
    'no_air_motion',
    'closed_odor_fluct',
    'white_odor_fluct',
]

EXPTS_ASENSORY = [
    'asensory_5_hz',
    'asensory_10_hz',
]

EXPTS_SENSORY = [
    'closed',
    'sinusoidal',
    'closed_white',
    'no_air_motion',
    'closed_odor_fluct',
    'white_odor_fluct',
]

EXPTS_W_AIR = [
    'sinusoidal',
    'closed_white',
    'closed_odor_fluct',
    'white_odor_fluct',
]

EXPTS_ODOR_FLUCT = [
    'closed_odor_fluct',
    'white_odor_fluct',
]

FILE_ENDINGS_ASENSORY = {
    'BEHAV': '.dat',
    'GCAMP': 'GCaMP_Signal.csv',
}

FILE_ENDINGS_SENSORY = {
    'BEHAV': '.dat',
    'T_GCAMP': 'GCaMP_Time_Pre.csv',
    'LIGHT': 'Light_Times.xlsx',
    'GCAMP': 'ROI-profiles.txt',
    'AIR': 'Air_Tube_Motion.csv',
    'ODOR_BINARY': 'Olfactometer.csv',
    'ODOR_PID': 'PID.csv',
}

COLS_FICTRAC = {
    'FRAME_CTR': 0,
    'V_LAT': 5,
    'V_FWD': 6,
    'V_ANG': 7,
    'HEADING': 16,
    'TIMESTAMP': 21,
}

DT_FICTRAC = 1./60

# DT_AIR = 0.00204

PFX_CLEAN = 'clean'

N_COLS_BEHAV = 4

COLS_BEHAV = {'V_LAT': 0, 'V_FWD': 1, 'V_ANG': 2, 'HEADING': 3}

LIMS_ANG = (-180, 180)

COLS_FINAL = [
    ('TIME', 0),
    ('G2R_RED', 1),
    ('G3R_RED', 2),
    ('G4R_RED', 3),
    ('G5R_RED', 4),
    ('G2L_RED', 5),
    ('G3L_RED', 6),
    ('G4L_RED', 7),
    ('G5L_RED', 8),
    ('G2R_GREEN', 9),
    ('G3R_GREEN', 10),
    ('G4R_GREEN', 11),
    ('G5R_GREEN', 12),
    ('G2L_GREEN', 13),
    ('G3L_GREEN', 14),
    ('G4L_GREEN', 15),
    ('G5L_GREEN', 16),
    ('V_LAT', 17),
    ('V_FWD', 18),
    ('V_ANG', 19),
    ('HEADING', 20),
    ('AIR', 21),
    ('W_AIR', 22),
    ('ODOR_BINARY', 23),
    ('ODOR_PID', 24),
]

N_COLS_FINAL = len(COLS_FINAL)

COL_SLICE_GCAMP = slice(1, 17)
COL_SLICE_BEHAV = slice(17, 21)
COL_SLICE_AIR = slice(21, 23)
COL_SLICE_ODOR = slice(23, 25)

# EXPERIMENT FEATURES
WRAPPED_ANG_VARS = ['heading', 'air_tube']
ODOR_START = 90  # s
ODOR_END = 150  # s
ODOR_BINARY_CUTOFF = 2.5  # V
AIR_FLOW_MAX_ANGLE = 90  # deg
AIR_FLOW_OFF_ANGLE = 180 * 2.67 / PI

T_MAX = 300

DRIVEN_RANDOM_EPOCHS = [(60, 120), (180, 240)]

# LOADING PARAMETERS
# VELOCITY_FILTER_DEFAULT = {'AMP': 2, 'TAU': 0.8, 'T_MAX_FILT': 8}

DAN_ORDER = ['G2L', 'G3L', 'G4L', 'G5L', 'G2R', 'G3R', 'G4R', 'G5R']

COLORS = {
    'speed': (0, 0, 0),  # black
    'int_speed': (0, 0, 0),  # black
    'v_forward': (153/255, 0, 76/255),  # dark pink
    'int_v_forward': (153/255, 0, 76/255),  # dark pink
    'v_angular': (76/255, 0, 153/255),  # dark purple
    'wnm_v_angular': (76/255, 0, 153/255),  # dark purple
    'int_v_angular': (76/255, 0, 153/255),  # dark purple
    'heading': (102/255, 102/255, 0),  # dark yellow
    'ddt_heading': (102/255, 102/255, 0),  # dark yellow
    'air_tube': (204/255, 0, 102/255),  # dark pink
    'ddt_air_tube': (204/255, 0, 102/255),  # dark pink

    'G2L': (153/255, 0, 0),  # dark red
    'G2R': (1, 0, 0),  # red
    'G2S': (1, 102/255, 102/255),  # pasty red
    'G2D': (1, 180,255, 180/255),  # pastier red

    'G3L': (0, 0, 153/255),  # dark blue
    'G3R': (0, 0, 1),  # blue
    'G3S': (102/255, 102/255, 1),  # pasty blue
    'G3D': (180/255, 180/255, 1),  # pastier blue

    'G4L': (0, 102/255, 0),  # dark green
    'G4R': (0, 180/255, 0),  # green
    'G4S': (51/255, 1, 51/255),  # pasty green
    'G4D': (0, 180/255, 0),  # green

    'G5L': (153/255, 76/255, 0),  # dark orange
    'G5R': (1, 128/255, 0),  # orange
    'G5S': (1, 178/255, 102/255),  # pasty orange
    'G5D': (1, 204/255, 153/255),  # pastier orange

    'ddt_G2L': (153/255, 0, 0),  # dark red
    'ddt_G2R': (1, 0, 0),  # red
    'ddt_G2S': (1, 102/255, 102/255),  # pasty red
    'ddt_G2D': (1, 180,255, 180/255),  # pastier red

    'ddt_G3L': (0, 0, 153/255),  # dark blue
    'ddt_G3R': (0, 0, 1),  # blue
    'ddt_G3S': (102/255, 102/255, 1),  # pasty blue
    'ddt_G3D': (180/255, 180/255, 1),  # pastier blue

    'ddt_G4L': (0, 102/255, 0),  # dark green
    'ddt_G4R': (0, 180/255, 0),  # green
    'ddt_G4S': (51/255, 1, 51/255),  # pasty green
    'ddt_G4D': (0, 0, 1),  # blue

    'ddt_G5L': (153/255, 76/255, 0),  # dark orange
    'ddt_G5R': (1, 128/255, 0),  # orange
    'ddt_G5S': (1, 178/255, 102/255),  # pasty orange
    'ddt_G5D': (1, 204/255, 153/255),  # pastier orange
}