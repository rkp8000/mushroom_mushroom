"""Some constants for this project."""
# FILE NAMES
BEHAV_FILE_ENDING = '.dat'
GCAMP_FILE = 'GCaMP_Time_Pre.csv'
LIGHT_TIMES_FILE = 'Light_Times.xlsx'
ROI_PROFILES_FILE = 'ROI-profiles.txt'
AIR_TUBE_FILE = 'Air_Tube_Motion.csv'


# LOADING PARAMETERS
VELOCITY_FILTER_DEFAULT = {'AMP': 2, 'TAU': 0.8, 'T_MAX_FILT': 8}
MAX_TRIAL_TIME = 301

DAN_ORDER = ['G2L', 'G3L', 'G4L', 'G5L', 'G2R', 'G3R', 'G4R', 'G5R']

COLORS_DEFAULT = {
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