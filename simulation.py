import math
import random
import copy # For deepcopy
import config # Keep for fallback or if current_tote_config is not passed in some contexts
import core_utils

def attempt_place_case(case_obj, tote_obj):
    """
    Tries to find the best valid placement for the case in the tote for simulation,
    prioritizing the lowest possible Z-level.
    """
    best_placement_info = {
        "can_fit": False,
        "chosen_orientation_dims": None,
        "position_in_tote_grid": None,
        "placement_z_level": float('inf')
    }

    for oriented_dims in case_obj["orientations"]:
        case_L_oriented, case_W_oriented, case_H_oriented = oriented_dims
        case_grid_L = max(1, math.ceil(case_L_oriented / tote_obj["height_map_resolution"]))
        case_grid_W = max(1, math.ceil(case_W_oriented / tote_obj["height_map_resolution"]))

        for start_grid_y in range(tote_obj["grid_dim_y"] - case_grid_W + 1):
            for start_grid_x in range(tote_obj["grid_dim_x"] - case_grid_L + 1):
                current_placement_base_z = 0.0
                footprint_valid = True
                for R_offset in range(case_grid_W):
                    for C_offset in range(case_grid_L):
                        current_grid_x = start_grid_x + C_offset
                        current_grid_y = start_grid_y + R_offset
                        if not (0 <= current_grid_x < tote_obj["grid_dim_x"] and \
                                0 <= current_grid_y < tote_obj["grid_dim_y"]):
                            footprint_valid = False; break
                        current_placement_base_z = max(current_placement_base_z, tote_obj["height_map"][current_grid_x][current_grid_y])
                    if not footprint_valid: break
                if not footprint_valid: continue

                if current_placement_base_z + case_H_oriented <= tote_obj["max_height"]:
                    if current_placement_base_z < best_placement_info["placement_z_level"]:
                        best_placement_info = {
                            "can_fit": True,
                            "chosen_orientation_dims": oriented_dims,
                            "position_in_tote_grid": (start_grid_x, start_grid_y),
                            "placement_z_level": current_placement_base_z
                        }
    return best_placement_info

def add_case_to_tote_and_update_state(case_obj, tote_obj, placement_details):
    """Adds case to tote, updates simulation state, and records interim utilization."""
    case_obj["chosen_orientation_dims"] = placement_details["chosen_orientation_dims"]
    case_obj["position_in_tote_grid"] = placement_details["position_in_tote_grid"]
    case_obj["placement_z_level"] = placement_details["placement_z_level"]

    oriented_L, oriented_W, oriented_H = case_obj["chosen_orientation_dims"]
    start_grid_x, start_grid_y = case_obj["position_in_tote_grid"]
    new_surface_height = case_obj["placement_z_level"] + oriented_H

    case_grid_L = max(1, math.ceil(oriented_L / tote_obj["height_map_resolution"]))
    case_grid_W = max(1, math.ceil(oriented_W / tote_obj["height_map_resolution"]))

    for R_offset in range(case_grid_W):
        for C_offset in range(case_grid_L):
            current_grid_x = start_grid_x + C_offset
            current_grid_y = start_grid_y + R_offset
            if 0 <= current_grid_x < tote_obj["grid_dim_x"] and \
               0 <= current_grid_y < tote_obj["grid_dim_y"]:
                tote_obj["height_map"][current_grid_x][current_grid_y] = new_surface_height

    tote_obj["items"].append(case_obj) # Add case to items list
    tote_obj["remaining_volume"] -= case_obj["volume"] # Update remaining volume

    # Calculate and store interim utilization on the case object itself
    utilized_volume_now = tote_obj["max_volume"] - tote_obj["remaining_volume"]
    current_util_percent_now = 0.0
    if tote_obj["max_volume"] > 0 and len(tote_obj["items"]) > 0:
        current_util_percent_now = (utilized_volume_now / tote_obj["max_volume"]) * 100.0
    case_obj["interim_tote_utilization_at_placement"] = current_util_percent_now

def finalize_and_store_tote(tote_obj, target_list):
    """Finalizes a tote (calculates final utilization) and stores its data for the simulation."""
    if tote_obj["max_volume"] > 0 and len(tote_obj["items"]) > 0:
        utilized_volume = tote_obj["max_volume"] - tote_obj["remaining_volume"]
        tote_obj["utilization_percent"] = (utilized_volume / tote_obj["max_volume"]) * 100.0
    else:
        tote_obj["utilization_percent"] = 0.0 # For empty totes
    target_list.append(copy.deepcopy(tote_obj))

def generate_test_cases(num_cases, seed=None, current_tote_config=None): # Added current_tote_config
    """Generates random case data for the simulation, respecting current tote dimensions if provided."""
    if seed is not None:
        random.seed(seed)
    test_cases_data = []

    if current_tote_config:
        max_l_bound = current_tote_config["TOTE_MAX_LENGTH"] - 100
        max_w_bound = current_tote_config["TOTE_MAX_WIDTH"] - 100
        max_h_bound = current_tote_config["TOTE_MAX_HEIGHT"] - 150
        h_map_res = current_tote_config["HEIGHT_MAP_RESOLUTION"]
    else: # Fallback to global config if no specific config is passed
        max_l_bound = config.TOTE_MAX_LENGTH - 100
        max_w_bound = config.TOTE_MAX_WIDTH - 100
        max_h_bound = config.TOTE_MAX_HEIGHT - 150
        h_map_res = config.HEIGHT_MAP_RESOLUTION

    for i in range(num_cases):
        # Ensure bounds are positive
        l = random.randint(50, max(51, max_l_bound))
        w = random.randint(50, max(51, max_w_bound))
        h = random.randint(50, max(51, max_h_bound))

        # Ensure dimensions are at least twice the resolution
        l = max(l, 2 * h_map_res)
        w = max(w, 2 * h_map_res)
        h = max(h, 2 * h_map_res)
        test_cases_data.append({"sku": f"SKU{i+1:03}", "length": l, "width": w, "height": h})
    return test_cases_data

def run_simulation_for_visualization_data(case_data_list, current_tote_config): # Added current_tote_config
    """
    Runs the packing simulation as a generator, yielding progress and intermediate results.
    Uses provided current_tote_config for tote dimensions and properties.
    Yields dictionaries with 'progress', 'status_message', 'intermediate_totes_data',
    'intermediate_vis_data', and 'unplaceable_log'.
    """
    all_processed_totes_full_data = []
    visualization_output_list = []
    unplaceable_cases_log = []
    total_cases = len(case_data_list)
    processed_cases = 0

    next_tote_id = 1
    # Use current_tote_config for creating totes
    current_tote = core_utils.create_new_empty_tote(next_tote_id, current_tote_config)

    status_message = f"Starting simulation with {total_cases} cases..."
    print(status_message) # Keep console log for debugging if needed
    yield {
        "progress": 0.0,
        "status_message": status_message,
        "intermediate_totes_data": [],
        "intermediate_vis_data": [],
        "unplaceable_log": []
    }

    for i, case_raw_data in enumerate(case_data_list):
        processed_cases = i + 1
        progress = processed_cases / total_cases if total_cases > 0 else 0.0
        status_message = f"Processing case {processed_cases}/{total_cases} (SKU: {case_raw_data.get('sku', 'N/A')})..."

        current_case = core_utils.get_case_properties(
            case_raw_data["sku"],
            case_raw_data["length"],
            case_raw_data["width"],
            case_raw_data["height"]
        )
        # Use current_tote_config for checks
        is_fundamentally_too_large_vol = current_case["volume"] > current_tote_config["TOTE_MAX_VOLUME"]
        can_orient_to_fit_empty_tote_dims = any(
            l_o <= current_tote_config["TOTE_MAX_LENGTH"] and \
            w_o <= current_tote_config["TOTE_MAX_WIDTH"] and \
            h_o <= current_tote_config["TOTE_MAX_HEIGHT"]
            for l_o, w_o, h_o in current_case["orientations"]
        )
        is_fundamentally_too_large_dims = not can_orient_to_fit_empty_tote_dims

        if is_fundamentally_too_large_vol or is_fundamentally_too_large_dims:
            reason = "volume" if is_fundamentally_too_large_vol else "dimensions"
            status_message = f"Case SKU {current_case['sku']} is fundamentally too large ({reason}). Skipping."
            print(f"  LOG: {status_message}")
            unplaceable_cases_log.append({"sku": current_case['sku'], "reason": f"Fundamentally too large ({reason})"})
            # Yield progress even when skipping
            yield {
                "progress": progress,
                "status_message": status_message,
                "intermediate_totes_data": copy.deepcopy(all_processed_totes_full_data + ([current_tote] if current_tote else [])), # Include current tote state
                "intermediate_vis_data": copy.deepcopy(visualization_output_list),
                "unplaceable_log": copy.deepcopy(unplaceable_cases_log)
            }
            continue

        # attempt_place_case internally uses tote_obj's dimensions
        placement_details = attempt_place_case(current_case, current_tote)
        can_fit_volumetrically = current_case["volume"] <= current_tote["remaining_volume"]

        if placement_details["can_fit"] and can_fit_volumetrically:
            add_case_to_tote_and_update_state(current_case, current_tote, placement_details)
            status_message = f"Placed {current_case['sku']} in Tote {current_tote['id']} (Util: {current_case['interim_tote_utilization_at_placement']:.1f}%)"
            print(f"  SUCCESS: {status_message}")
        else:
            fit_reason = "no spatial fit" if not placement_details["can_fit"] else "insufficient remaining volume"
            status_message = f"Case {current_case['sku']} doesn't fit Tote {current_tote['id']} ({fit_reason}). Finalizing tote."
            print(f"  INFO: {status_message}")
            finalize_and_store_tote(current_tote, all_processed_totes_full_data)

            next_tote_id += 1
            # Use current_tote_config for new totes
            current_tote = core_utils.create_new_empty_tote(next_tote_id, current_tote_config)
            status_message = f"Started new Tote {current_tote['id']}."
            print(f"  INFO: {status_message}")

            placement_details_new_tote = attempt_place_case(current_case, current_tote)
            can_fit_new_tote_volumetrically = current_case["volume"] <= current_tote["remaining_volume"]

            if placement_details_new_tote["can_fit"] and can_fit_new_tote_volumetrically:
                add_case_to_tote_and_update_state(current_case, current_tote, placement_details_new_tote)
                status_message = f"Placed {current_case['sku']} in new Tote {current_tote['id']} (Util: {current_case['interim_tote_utilization_at_placement']:.1f}%)"
                print(f"  SUCCESS: {status_message}")
            else:
                reason_new_tote = "no spatial fit" if not placement_details_new_tote["can_fit"] else "insufficient volume (unexpected)"
                status_message = f"Case SKU {current_case['sku']} could not be placed even in new Tote {current_tote['id']} ({reason_new_tote}). Skipping."
                print(f"  ERROR: {status_message}")
                unplaceable_cases_log.append({"sku": current_case['sku'], "reason": f"Could not fit new empty tote ({reason_new_tote})"})

        # --- Prepare data for visualization list (do this incrementally) ---
        # Find the tote the item was *actually* placed in (could be current_tote or the last one in all_processed_totes_full_data if a new one was just started)
        placed_in_tote = None
        if "position_in_tote_grid" in current_case: # Check if the case was successfully placed in *some* tote
            if current_case in current_tote["items"]:
                 placed_in_tote = current_tote
            # This check might be complex if the item caused the *previous* tote to finalize.
            # For simplicity, we'll rebuild the vis_list from the tote data each yield.

        # --- Rebuild visualization list based on current state ---
        current_visualization_output_list = []
        # Include finalized totes
        for tote in all_processed_totes_full_data:
            for item in tote["items"]:
                grid_x, grid_y = item["position_in_tote_grid"]
                actual_x_coord = grid_x * tote["height_map_resolution"]
                actual_y_coord = grid_y * tote["height_map_resolution"]
                actual_z_coord = item["placement_z_level"]
                current_visualization_output_list.append({
                    "tote_id": tote["id"],
                    "tote_dimensions_mm": {"length": tote["max_length"], "width": tote["max_width"], "height": tote["max_height"]},
                    "case_sku": item["sku"],
                    "original_case_dims_mm": item["original_dims"],
                    "placed_case_dims_mm": {"length": item["chosen_orientation_dims"][0], "width": item["chosen_orientation_dims"][1], "height": item["chosen_orientation_dims"][2]},
                    "position_mm": { "x": actual_x_coord, "y": actual_y_coord, "z": actual_z_coord },
                    "current_tote_utilization_percent": item.get("interim_tote_utilization_at_placement", 0.0)
                })
        # Include items in the currently active tote
        if current_tote:
            for item in current_tote["items"]:
                 grid_x, grid_y = item["position_in_tote_grid"]
                 actual_x_coord = grid_x * current_tote["height_map_resolution"]
                 actual_y_coord = grid_y * current_tote["height_map_resolution"]
                 actual_z_coord = item["placement_z_level"]
                 current_visualization_output_list.append({
                    "tote_id": current_tote["id"],
                    "tote_dimensions_mm": {"length": current_tote["max_length"], "width": current_tote["max_width"], "height": current_tote["max_height"]},
                    "case_sku": item["sku"],
                    "original_case_dims_mm": item["original_dims"],
                    "placed_case_dims_mm": {"length": item["chosen_orientation_dims"][0], "width": item["chosen_orientation_dims"][1], "height": item["chosen_orientation_dims"][2]},
                    "position_mm": { "x": actual_x_coord, "y": actual_y_coord, "z": actual_z_coord },
                    "current_tote_utilization_percent": item.get("interim_tote_utilization_at_placement", 0.0)
                 })


        # Yield current state
        yield {
            "progress": progress,
            "status_message": status_message,
            # Combine finalized totes with the current active tote for intermediate display
            "intermediate_totes_data": copy.deepcopy(all_processed_totes_full_data + ([current_tote] if current_tote else [])),
            "intermediate_vis_data": copy.deepcopy(current_visualization_output_list), # Use the rebuilt list
            "unplaceable_log": copy.deepcopy(unplaceable_cases_log)
        }


    # --- Finalization after loop ---
    if current_tote and (len(current_tote["items"]) > 0 or (current_tote["id"] == 1 and not all_processed_totes_full_data)):
        finalize_and_store_tote(current_tote, all_processed_totes_full_data)
        status_message = f"Finalizing last active Tote {current_tote['id']} (Util: {current_tote['utilization_percent']:.1f}%)"
        print(f"\n{status_message}")
    else:
        status_message = "Simulation loop finished."
        print(f"\n{status_message}")


    # --- Final Yield (or could return this) ---
    # Rebuild final visualization list one last time
    final_visualization_output_list = []
    for tote in all_processed_totes_full_data:
        for item in tote["items"]:
            grid_x, grid_y = item["position_in_tote_grid"]
            actual_x_coord = grid_x * tote["height_map_resolution"]
            actual_y_coord = grid_y * tote["height_map_resolution"]
            actual_z_coord = item["placement_z_level"]
            final_visualization_output_list.append({
                "tote_id": tote["id"],
                "tote_dimensions_mm": {"length": tote["max_length"], "width": tote["max_width"], "height": tote["max_height"]},
                "case_sku": item["sku"],
                "original_case_dims_mm": item["original_dims"],
                "placed_case_dims_mm": {"length": item["chosen_orientation_dims"][0], "width": item["chosen_orientation_dims"][1], "height": item["chosen_orientation_dims"][2]},
                "position_mm": { "x": actual_x_coord, "y": actual_y_coord, "z": actual_z_coord },
                "current_tote_utilization_percent": item.get("interim_tote_utilization_at_placement", 0.0)
            })

    final_status = f"Simulation finished. Processed {len(all_processed_totes_full_data)} totes. Placed: {len(final_visualization_output_list)}. Unplaceable: {len(unplaceable_cases_log)}."
    print(final_status)
    if unplaceable_cases_log:
         print("Unplaceable items:")
         for entry in unplaceable_cases_log:
              print(f"  - SKU: {entry['sku']}, Reason: {entry['reason']}")

    yield {
        "progress": 1.0,
        "status_message": final_status,
        "intermediate_totes_data": copy.deepcopy(all_processed_totes_full_data), # Final tote data
        "intermediate_vis_data": copy.deepcopy(final_visualization_output_list), # Final vis data
        "unplaceable_log": copy.deepcopy(unplaceable_cases_log),
        "is_final": True # Add a flag to indicate completion
    }

    # The generator naturally stops here after the final yield.
    # No explicit return needed unless you want to return something *after* the last yield,
    # but the final state is captured in the last yielded dictionary.
