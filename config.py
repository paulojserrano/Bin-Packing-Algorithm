import math

# --- Global Configuration ---
TOTE_MAX_LENGTH = round(31.5*25.4)
TOTE_MAX_WIDTH = round(24*25.4)
TOTE_MAX_HEIGHT = round(22.4*25.4)
TOTE_MAX_VOLUME = TOTE_MAX_LENGTH * TOTE_MAX_WIDTH * TOTE_MAX_HEIGHT
HEIGHT_MAP_RESOLUTION = 20 # mm

GRID_DIM_X = math.ceil(TOTE_MAX_LENGTH / HEIGHT_MAP_RESOLUTION)
GRID_DIM_Y = math.ceil(TOTE_MAX_WIDTH / HEIGHT_MAP_RESOLUTION)

# --- Default values for new parameters ---
DEFAULT_MAX_WEIGHT_PER_TOTE = 25.0 # e.g., kg or lbs (user to specify unit in UI)
DEFAULT_MAX_UNIQUE_SKUS_PER_TOTE = 3
