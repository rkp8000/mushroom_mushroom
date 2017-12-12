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

# COLOR SCHEME
COLORS_RGB = {
    # G2
    'g2r': (0, 0, 153),  # dark blue
    'g2l': (0, 0, 255),  # light blue
    'g2s': (0, 0, 153),  # dark blue
    'g2d': (0, 0, 255),  # light blue
    
    'ddt_g2r': (0, 0, 153),  # dark blue
    'ddt_g2l': (0, 0, 255),  # light blue
    'ddt_g2s': (0, 0, 153),  # dark blue
    'ddt_g2d': (0, 0, 255),  # light blue
    
    # G3
    'g3r': (153, 0, 76),  # dark pink
    'g3l': (255, 0, 127),  # light pink
    'g3s': (153, 0, 76),  # dark pink
    'g3d': (255, 0, 127),  # light pink
    
    'ddt_g3r': (153, 0, 76),  # dark pink
    'ddt_g3l': (255, 0, 127),  # light pink
    'ddt_g3s': (153, 0, 76),  # dark pink
    'ddt_g3d': (255, 0, 127),  # light pink
    
    # G4
    'g4r': (0, 102, 0),  # dark green
    'g4l': (0, 204, 0),  # light green
    'g4s': (0, 102, 0),  # dark green
    'g4d': (0, 204, 0),  # light green
    
    'ddt_g4r': (0, 102, 0),  # dark green
    'ddt_g4l': (0, 204, 0),  # light green
    'ddt_g4s': (0, 102, 0),  # dark green
    'ddt_g4d': (0, 204, 0),  # light green
    
    # G5
    'g5r': (76, 0, 153),  # dark violet
    'g5l': (127, 0, 255),  # light violet
    'g5s': (76, 0, 153),  # dark violet
    'g5d': (127, 0, 255),  # light violet
    
    'ddt_g5r': (76, 0, 153),  # dark violet
    'ddt_g5l': (127, 0, 255),  # light violet
    'ddt_g5s': (76, 0, 153),  # dark violet
    'ddt_g5d': (127, 0, 255),  # light violet
    
    # BEHAV
    'ball': (0, 0, 0),  # black
    'speed': (96, 96, 96),  # dark gray
    'v_fwd': (160, 160, 160),  # light gray
    'v_lat': (153, 76, 0),  # dark orange
    'v_ang': (255, 128, 0),  # light orange
    'heading': (0, 204, 204),  # cyan
    'ddt_heading': (0, 204, 204),  # cyan
    
    # SENSORY
    'air': (0, 0, 0),  # black
    'w_air': (0, 0, 0),  # black
    
    'odor_binary': (153, 0, 0),  # dark red
    'odor_pid': (255, 0, 0),  # light red
}