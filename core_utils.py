import config

def create_new_empty_tote(tote_id):
    """Initializes a new, empty tote object for the simulation."""
    new_tote = {
        "id": tote_id, "max_length": config.TOTE_MAX_LENGTH, "max_width": config.TOTE_MAX_WIDTH,
        "max_height": config.TOTE_MAX_HEIGHT, "max_volume": config.TOTE_MAX_VOLUME,
        "grid_dim_x": config.GRID_DIM_X, "grid_dim_y": config.GRID_DIM_Y,
        "height_map_resolution": config.HEIGHT_MAP_RESOLUTION, "items": [],
        "remaining_volume": config.TOTE_MAX_VOLUME,
        "height_map": [[0.0 for _ in range(config.GRID_DIM_Y)] for _ in range(config.GRID_DIM_X)],
        "utilization_percent": 0.0 # Final utilization
    }
    return new_tote

def get_case_properties(sku, length, width, height):
    """Creates a case object with its properties and orientations for the simulation."""
    volume = length * width * height
    orientations = [
        (length, width, height), (length, height, width),
        (width, length, height), (width, height, length),
        (height, length, width), (height, width, length)
    ]
    return {
        "sku": sku, "original_dims": (length, width, height), "volume": volume,
        "orientations": orientations, "chosen_orientation_dims": None,
        "position_in_tote_grid": None, "placement_z_level": None,
        "interim_tote_utilization_at_placement": 0.0 # New field
    }
