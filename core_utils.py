# import config # This import is no longer strictly needed for tote dimensions in create_new_empty_tote

def create_new_empty_tote(tote_id, tote_config, max_weight_per_tote, max_unique_skus_per_tote): # Added tote_config parameter
    """Initializes a new, empty tote object for the simulation using provided config."""
    new_tote = {
        "id": tote_id,
        "max_length": tote_config["TOTE_MAX_LENGTH"], # Use tote_config
        "max_width": tote_config["TOTE_MAX_WIDTH"],   # Use tote_config
        "max_height": tote_config["TOTE_MAX_HEIGHT"], # Use tote_config
        "max_volume": tote_config["TOTE_MAX_VOLUME"], # Use tote_config
        "grid_dim_x": tote_config["GRID_DIM_X"],       # Use tote_config
        "grid_dim_y": tote_config["GRID_DIM_Y"],       # Use tote_config
        "height_map_resolution": tote_config["HEIGHT_MAP_RESOLUTION"], # Use tote_config
        "items": [],
        "remaining_volume": tote_config["TOTE_MAX_VOLUME"], # Use tote_config
        "height_map": [[0.0 for _ in range(tote_config["GRID_DIM_Y"])] for _ in range(tote_config["GRID_DIM_X"])],
        "max_weight": max_weight_per_tote, # Add this
        "current_weight": 0.0, # Add this
        "max_unique_skus": max_unique_skus_per_tote, # Add this
        "current_skus": set(), # Add this
        "utilization_percent": 0.0
    }
    return new_tote

def get_case_properties(sku, length, width, height, weight=0): # Add weight, default to 0
    """Creates a case object with its properties and orientations for the simulation."""
    volume = length * width * height
    orientations = [
        (length, width, height), (length, height, width),
        (width, length, height), (width, height, length),
        (height, length, width), (height, width, length)
    ]
    # Ensure all dimensions are positive
    orientations = [
        tuple(max(1, dim_val) for dim_val in o) for o in orientations
    ]
    return {
        "sku": sku, "original_dims": (max(1,length), max(1,width), max(1,height)), "volume": max(1,volume),
        "weight": weight, # Add this line
        "orientations": orientations, "chosen_orientation_dims": None,
        "position_in_tote_grid": None, "placement_z_level": None,
        "interim_tote_utilization_at_placement": 0.0
    }
