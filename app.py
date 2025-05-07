# app.py
import streamlit as st
import pandas as pd
import math # For calculations
import matplotlib.pyplot as plt # Added for plt.close() and histogram
import statistics # For median
import numpy as np # For histogram binning and other calculations if needed
from collections import Counter, defaultdict # For new statistics

# Import existing modules from the project
import simulation
import core_utils
import visualization # Now includes generate_tote_figure
import config # To get default values initially

# --- UI Configuration ---
st.set_page_config(layout="wide", page_title="Bin Packing Simulation")
st.title("Interactive Bin Packing Simulation")

# --- Algorithm Explanation (Collapsible) ---
with st.expander("Algorithm Overview and Methodology", expanded=False):
    st.markdown("""
    This simulation employs a heuristic approach to address the three-dimensional bin packing problem (3D-BPP). The 3D-BPP is a classic NP-hard problem, meaning that finding a guaranteed optimal solution is computationally infeasible for all but the smallest instances. Therefore, heuristic methods are commonly used to find good, practical solutions in a reasonable timeframe.

    **Core Algorithm:**

    The algorithm implemented here can be characterized as a **greedy, single-pass, sequential heuristic with item orientation and height-map utilization.** It processes items one by one in the order they are provided:

    1.  **Item Consideration:** For each incoming item (case), the algorithm evaluates a set of predefined orthogonal orientations (typically up to six, by permuting length, width, and height).
    2.  **Placement Strategy (Height Map):**
        * The available space within a tote is represented by a 2D grid (height map) corresponding to the tote's base. Each cell in this grid stores the current maximum height of packed items at that (x, y) location.
        * For each orientation of the current item, the algorithm attempts to find a valid placement position on this grid. It iterates through all possible (x, y) starting positions on the grid.
        * A placement is considered valid if the item's footprint does not exceed the tote's boundaries and if the height of the item, when placed on the highest point within its footprint on the height map, does not exceed the tote's maximum height.
        * The algorithm prioritizes placements at the **lowest possible Z-level (height)**. Among positions offering the same minimal Z-level, the first one encountered is typically chosen.
    3.  **Tote Management:**
        * Items are placed into the currently active tote if a valid position is found.
        * If an item cannot be placed in the current tote (either due to spatial constraints or volumetric capacity), the current tote is considered finalized, and a new, empty tote is initiated. The algorithm then attempts to place the item in this new tote.
        * If an item cannot fit even into a new empty tote (e.g., its dimensions exceed the tote's dimensions in all orientations), it is marked as unplaceable.
    4.  **Output:** The simulation provides data on the number of totes used, the items packed within each tote, their specific positions and orientations, and the volumetric utilization of each tote.

    **Limitations of the Implemented Heuristic:**

    While this greedy heuristic is computationally efficient and often provides reasonable packing solutions, it is important to acknowledge its inherent limitations:

    * **Non-Optimal Solutions:** Being a heuristic, the algorithm does not guarantee a globally optimal solution. The chosen placement for an item is based on local criteria at that specific point in the packing sequence and may prevent more efficient placements later on. This can result in using more totes or achieving lower overall density than theoretically possible.
    * **Order Dependency:** The sequence in which items are presented to the algorithm can significantly impact the packing outcome. Different input orders for the same set of items may yield different results in terms of tote count and utilization. Pre-sorting items (e.g., by volume, largest dimension, or other criteria – a common strategy in algorithms like First Fit Decreasing) is not explicitly implemented here but could be a pre-processing step to potentially improve results.
    * **Discretization Effects (Height Map Resolution):** The use of a height map with a defined resolution means that the available space is discretized. This can lead to:
        * **Internal Fragmentation:** Small, unusable spaces may be created if item dimensions do not align perfectly with the grid resolution.
        * **Approximation:** The algorithm effectively treats items as if their base dimensions are multiples of the resolution for placement checking, which might lead to slightly suboptimal fits for items with dimensions not aligning with the grid. A finer resolution can improve accuracy but increases computational cost.
    * **Limited Lookahead & No Backtracking:** The algorithm makes a decision for each item and does not revisit previously placed items to rearrange them for a better global fit (no backtracking). It lacks a "lookahead" capability to anticipate how current placements might affect future possibilities.
    * **Rotation Constraints:** Only orthogonal rotations (permutations of L, W, H) are considered. Arbitrary rotations are not evaluated, which could be beneficial for irregularly shaped items or achieving tighter fits in some scenarios, albeit at a much higher computational cost.
    * **Single Tote Focus at a Time:** The decision to open a new tote is made only when an item cannot fit into the current one. More advanced strategies might consider distributing items across multiple partially filled totes simultaneously.

    Despite these limitations, the implemented approach provides a valuable tool for estimating packing configurations and understanding the complexities of 3D bin packing in practical scenarios.
    """)
st.divider() # Adds a visual separator

# --- Initialize Session State ---
# For CSV column mapping
if 'csv_headers' not in st.session_state:
    st.session_state.csv_headers = []
if 'column_mappings' not in st.session_state:
    st.session_state.column_mappings = {
        'length_col': None, 'width_col': None, 'height_col': None, 'sku_col': None
    }
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None

# For simulation results
if 'simulation_ran' not in st.session_state:
    st.session_state.simulation_ran = False
if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = {
        'visualization_output_list': [],
        'full_totes_summary_data': []
    }
if 'original_item_count' not in st.session_state: # To store original item count for captions
    st.session_state.original_item_count = 0


# --- Sidebar Configuration ---
st.sidebar.header("Simulation Configuration")

# --- Tote Configuration ---
st.sidebar.subheader("Tote Dimensions (mm)")
tote_length_input = st.sidebar.number_input(
    "Tote Length", min_value=50, value=int(config.TOTE_MAX_LENGTH), step=10, key="tote_length" # Cast to int for number_input
)
tote_width_input = st.sidebar.number_input(
    "Tote Width", min_value=50, value=int(config.TOTE_MAX_WIDTH), step=10, key="tote_width" # Cast to int
)
tote_height_input = st.sidebar.number_input(
    "Tote Height", min_value=50, value=int(config.TOTE_MAX_HEIGHT), step=10, key="tote_height" # Cast to int
)
height_map_resolution_input = st.sidebar.number_input(
    "Height Map Resolution (mm)", min_value=1, value=config.HEIGHT_MAP_RESOLUTION, step=1, key="height_map_resolution"
)

# --- Case Generation ---
st.sidebar.subheader("Case Data Source")
# When data source changes, reset simulation_ran to ensure results are cleared if user modifies source then reruns
def reset_sim_ran_on_source_change():
    """Resets simulation state when the data source (random/CSV) is changed."""
    st.session_state.simulation_ran = False
    # Also clear previous results to avoid showing stale data if user switches source then reruns old config
    st.session_state.simulation_results = {'visualization_output_list': [], 'full_totes_summary_data': []}
    st.session_state.original_item_count = 0


case_data_source = st.sidebar.radio(
    "Select case data source:",
    ("Generate Random Cases", "Upload CSV File"),
    key="case_data_source",
    on_change=reset_sim_ran_on_source_change
)

if case_data_source == "Generate Random Cases":
    # If switching from CSV to Random, clear CSV-related session state
    if st.session_state.uploaded_file_name is not None:
        st.session_state.csv_headers = []
        st.session_state.column_mappings = {k: None for k in st.session_state.column_mappings}
        st.session_state.uploaded_file_name = None

    st.sidebar.subheader("Random Case Generation")
    num_random_cases_input = st.sidebar.number_input(
        "Number of Random Cases", min_value=1, value=10, step=1, key="num_cases"
    )
    random_seed_input = st.sidebar.number_input(
        "Random Seed", value=42, step=1, key="random_seed"
    )
else: # "Upload CSV File"
    st.sidebar.subheader("Upload Case Data")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Case Data CSV", type=["csv"], key="case_csv_uploader"
    )

    if uploaded_file is not None:
        if st.session_state.uploaded_file_name != uploaded_file.name:
            st.session_state.uploaded_file_name = uploaded_file.name
            try:
                # Read only the header row first to get column names for mapping
                df_headers = pd.read_csv(uploaded_file, nrows=0)
                st.session_state.csv_headers = df_headers.columns.tolist()
                st.session_state.column_mappings = {k: None for k in st.session_state.column_mappings}
                uploaded_file.seek(0) # Reset file pointer for next read
                st.sidebar.success(f"File '{uploaded_file.name}' loaded. Map columns below.")
            except Exception as e:
                st.sidebar.error(f"Error reading CSV headers: {e}")
                st.session_state.csv_headers = []
                st.session_state.uploaded_file_name = None
    else:
        if st.session_state.uploaded_file_name is not None: # If a file was previously loaded but now removed
            st.session_state.csv_headers = []
            st.session_state.column_mappings = {k: None for k in st.session_state.column_mappings}
            st.session_state.uploaded_file_name = None

    if st.session_state.csv_headers:
        st.sidebar.subheader("Map CSV Columns")
        def get_col_index(col_key, options_list, default_idx=0):
            prev_selection = st.session_state.column_mappings.get(col_key)
            if prev_selection and prev_selection in options_list:
                return options_list.index(prev_selection)
            try:
                return min(default_idx, len(options_list) -1) if options_list else 0
            except ValueError:
                return 0

        st.session_state.column_mappings['length_col'] = st.sidebar.selectbox(
            "Length Column:", options=st.session_state.csv_headers,
            index=get_col_index('length_col', st.session_state.csv_headers, 0), key="map_length"
        )
        st.session_state.column_mappings['width_col'] = st.sidebar.selectbox(
            "Width Column:", options=st.session_state.csv_headers,
            index=get_col_index('width_col', st.session_state.csv_headers, 1), key="map_width"
        )
        st.session_state.column_mappings['height_col'] = st.sidebar.selectbox(
            "Height Column:", options=st.session_state.csv_headers,
            index=get_col_index('height_col', st.session_state.csv_headers, 2), key="map_height"
        )
        sku_options = ["Auto-generate SKU"] + st.session_state.csv_headers
        st.session_state.column_mappings['sku_col'] = st.sidebar.selectbox(
            "SKU Column (Optional):", options=sku_options,
            index=get_col_index('sku_col', sku_options, 0), key="map_sku"
        )
    elif uploaded_file and not st.session_state.csv_headers: # If file uploaded but headers couldn't be read
        st.sidebar.warning("Could not read columns. Check CSV format or re-upload.")


# --- Run Simulation Button ---
if st.sidebar.button("Run Simulation", key="run_button", type="primary"):
    st.session_state.simulation_ran = False
    st.session_state.simulation_results = {'visualization_output_list': [], 'full_totes_summary_data': []}
    st.session_state.original_item_count = 0 # Reset item count

    dynamic_tote_config = {
        "TOTE_MAX_LENGTH": int(tote_length_input),
        "TOTE_MAX_WIDTH": int(tote_width_input),
        "TOTE_MAX_HEIGHT": int(tote_height_input),
        "TOTE_MAX_VOLUME": int(tote_length_input * tote_width_input * tote_height_input),
        "HEIGHT_MAP_RESOLUTION": int(height_map_resolution_input),
        "GRID_DIM_X": max(1, math.ceil(int(tote_length_input) / int(height_map_resolution_input))),
        "GRID_DIM_Y": max(1, math.ceil(int(tote_width_input) / int(height_map_resolution_input)))
    }

    simulation_can_proceed = False
    current_input_cases = []

    if case_data_source == "Generate Random Cases":
        with st.spinner("Generating test cases..."):
            current_input_cases = simulation.generate_test_cases(
                num_cases=int(num_random_cases_input),
                seed=int(random_seed_input),
                current_tote_config=dynamic_tote_config
            )
        if current_input_cases:
            st.session_state.original_item_count = len(current_input_cases)
            simulation_can_proceed = True
        else:
            st.warning("No random cases were generated.")

    elif case_data_source == "Upload CSV File":
        if uploaded_file is not None:
            len_col = st.session_state.column_mappings.get('length_col')
            wid_col = st.session_state.column_mappings.get('width_col')
            hei_col = st.session_state.column_mappings.get('height_col')
            sku_col_map = st.session_state.column_mappings.get('sku_col')

            if not all([len_col, wid_col, hei_col]):
                st.error("Column mapping incomplete. Please select columns for Length, Width, and Height in the sidebar.")
            else:
                try:
                    uploaded_file.seek(0) # Ensure reading from the start of the file

                    # Determine which columns to read from the CSV
                    columns_to_read = []
                    if len_col: columns_to_read.append(len_col)
                    if wid_col: columns_to_read.append(wid_col)
                    if hei_col: columns_to_read.append(hei_col)
                    if sku_col_map and sku_col_map != "Auto-generate SKU":
                        columns_to_read.append(sku_col_map)
                    
                    # Remove duplicates in case a column was mapped to multiple roles (unlikely but safe)
                    columns_to_read = list(set(columns_to_read))

                    if not columns_to_read or not all(c in st.session_state.csv_headers for c in [len_col, wid_col, hei_col]):
                        st.error("One or more essential mapped columns (Length, Width, Height) are not in the CSV headers. Please check mappings.")
                    else:
                        # Read only the specified columns
                        df = pd.read_csv(uploaded_file, usecols=columns_to_read)
                    
                        # Validate data types and values for essential columns
                        required_mapped_cols = {'Length': len_col, 'Width': wid_col, 'Height': hei_col}
                        valid_data = True
                        for std_name, actual_col in required_mapped_cols.items():
                            if actual_col not in df.columns: # Should be caught by previous check, but good for safety
                                st.error(f"Mapped column '{actual_col}' for {std_name} not found in loaded CSV data. This should not happen.")
                                valid_data = False; break
                            if not pd.api.types.is_numeric_dtype(df[actual_col]):
                                st.error(f"Column '{actual_col}' (mapped to {std_name}) must contain numeric values.")
                                valid_data = False; break
                            if not (df[actual_col] > 0).all():
                                st.error(f"All values in column '{actual_col}' (mapped to {std_name}) must be positive.")
                                valid_data = False; break

                        if valid_data:
                            for index, row in df.iterrows():
                                sku_val = f"CSV_SKU_{index+1}"
                                if sku_col_map and sku_col_map != "Auto-generate SKU" and sku_col_map in df.columns:
                                    sku_val = str(row[sku_col_map])
                                # If sku_col_map was "Auto-generate SKU" or not in df.columns (e.g. not read), sku_val remains auto-generated

                                current_input_cases.append({
                                    "sku": sku_val,
                                    "length": float(row[len_col]),
                                    "width": float(row[wid_col]),
                                    "height": float(row[hei_col])
                                })
                            st.session_state.original_item_count = len(current_input_cases)
                            simulation_can_proceed = True
                except Exception as e:
                    st.error(f"Error processing CSV file with mapped columns: {e}")
        else:
            st.error("Please upload a CSV file when 'Upload CSV File' source is selected.")

    if simulation_can_proceed and current_input_cases:
        st.header("Simulation Process & Results")
        with st.spinner("Running packing simulation... This may take a moment."):
            sim_vis_data, full_totes_summary = simulation.run_simulation_for_visualization_data(
                case_data_list=current_input_cases,
                current_tote_config=dynamic_tote_config
            )
            st.session_state.simulation_results['visualization_output_list'] = sim_vis_data
            st.session_state.simulation_results['full_totes_summary_data'] = full_totes_summary
            st.session_state.simulation_ran = True
        st.success("Simulation finished!")

    elif simulation_can_proceed and not current_input_cases:
         st.warning("No case data was generated or loaded. Please check your inputs.")
    elif not simulation_can_proceed and (case_data_source == "Upload CSV File" and not uploaded_file and st.session_state.column_mappings.get('length_col')):
        pass # Avoid error if mappings exist but file is removed before run
    elif not simulation_can_proceed and current_input_cases: # Should be caught by specific errors above
        st.error("Simulation cannot proceed despite having case data. Check error messages above in the main panel or sidebar.")


# --- Display Results (Statistics first, then individual totes) ---
if st.session_state.simulation_ran:
    full_totes_summary_data = st.session_state.simulation_results['full_totes_summary_data']
    visualization_output_list = st.session_state.simulation_results['visualization_output_list']


    # --- Overall Statistics Section ---
    if full_totes_summary_data:
        st.subheader("Overall Simulation Statistics")

        # Pre-calculate all necessary values
        utilization_percentages = [
            tote.get('utilization_percent', 0.0)
            for tote in full_totes_summary_data
            if tote.get('utilization_percent') is not None
        ]
        total_totes_used = len(full_totes_summary_data)
        total_items_placed_in_stats = sum(len(tote.get('items',[])) for tote in full_totes_summary_data)
        unplaced_items_count = 0
        if 'original_item_count' in st.session_state and st.session_state.original_item_count >= total_items_placed_in_stats:
            unplaced_items_count = st.session_state.original_item_count - total_items_placed_in_stats
        
        avg_items_per_tote = (total_items_placed_in_stats / total_totes_used) if total_totes_used > 0 else 0.0
        totes_with_one_item = sum(1 for tote in full_totes_summary_data if len(tote.get('items', [])) == 1)
        percentage_single_case_totes = (totes_with_one_item / total_totes_used * 100) if total_totes_used > 0 else 0.0

        all_placed_skus_list = []
        sku_to_totes_map = defaultdict(set)
        for tote in full_totes_summary_data:
            tote_id = tote.get('id', 'UnknownTote')
            for item in tote.get('items', []):
                sku = item.get('sku', 'UnknownSKU')
                all_placed_skus_list.append(sku)
                sku_to_totes_map[sku].add(tote_id)
        num_unique_skus_placed = len(sku_to_totes_map)
        avg_totes_per_sku = 0.0
        if num_unique_skus_placed > 0:
            total_sku_presences_in_totes = sum(len(totes) for totes in sku_to_totes_map.values())
            avg_totes_per_sku = total_sku_presences_in_totes / num_unique_skus_placed
        most_frequent_sku_str = "N/A"
        if all_placed_skus_list:
            sku_counts = Counter(all_placed_skus_list)
            most_common_sku, most_common_count = sku_counts.most_common(1)[0]
            most_frequent_sku_str = f"{most_common_sku} (Count: {most_common_count})"

        # --- Group 1: Packing Efficiency & Tote Performance ---
        st.markdown("###### Packing Efficiency & Tote Performance")
        
        perf_data = [
            {"Metric": "Total Totes Used", "Value": str(total_totes_used)},
            {"Metric": "Total Cases Placed", "Value": str(total_items_placed_in_stats)},
            {"Metric": "Cases That Did Not Fit", "Value": str(unplaced_items_count)},
            {"Metric": "Average Items per Tote", "Value": f"{avg_items_per_tote:.2f}"},
            {"Metric": "Single-Case Totes (Count)", "Value": str(totes_with_one_item)},
            {"Metric": "Single-Case Totes (%)", "Value": f"{percentage_single_case_totes:.2f}%"},
        ]
        if utilization_percentages:
            perf_data.extend([
                {"Metric": "Average Tote Utilization", "Value": f"{statistics.mean(utilization_percentages):.2f}%"},
                {"Metric": "Minimum Tote Utilization", "Value": f"{min(utilization_percentages):.2f}%"},
                {"Metric": "Median Tote Utilization", "Value": f"{statistics.median(utilization_percentages):.2f}%"},
                {"Metric": "Maximum Tote Utilization", "Value": f"{max(utilization_percentages):.2f}%"},
            ])
        
        perf_df = pd.DataFrame(perf_data)
        st.table(perf_df.set_index('Metric')) # Using st.table for a cleaner look

        if utilization_percentages:
            st.markdown("###### Distribution of Tote Utilizations") # Sub-heading for histogram
            fig_hist, ax_hist = plt.subplots(figsize=(4.0, 2.25)) # Slightly larger for table context
            ax_hist.hist(utilization_percentages, bins='auto', color='skyblue', edgecolor='black')
            ax_hist.set_title('Tote Utilization %', fontsize=9)
            ax_hist.set_xlabel('Utilization %', fontsize=8)
            ax_hist.set_ylabel('No. of Totes', fontsize=8)
            ax_hist.tick_params(axis='both', which='major', labelsize=7)
            fig_hist.tight_layout(pad=0.5)
            st.pyplot(fig_hist, use_container_width=False) 
            plt.close(fig_hist)
        else:
            st.write("No utilization percentage data available for detailed utilization statistics or histogram.")

        # --- Group 2: SKU Distribution & Insights ---
        st.markdown("---") 
        st.markdown("###### SKU Distribution & Insights")
        
        sku_data = [
            {"Metric": "Total Unique SKUs Placed", "Value": str(num_unique_skus_placed)},
            {"Metric": "Avg. Totes per SKU", "Value": f"{avg_totes_per_sku:.2f}"},
            {"Metric": "Most Frequent SKU", "Value": most_frequent_sku_str},
        ]
        sku_df = pd.DataFrame(sku_data)
        st.table(sku_df.set_index('Metric'))

        # --- Export Report ---
        st.markdown("---")
        st.markdown("###### Export Full Pack Report")
        if visualization_output_list:
            report_data = []
            for item_vis_data in visualization_output_list:
                report_data.append({
                    "Tote ID": item_vis_data.get('tote_id', 'N/A'),
                    "SKU": item_vis_data.get('case_sku', 'N/A'),
                    "Placed Length (mm)": item_vis_data.get('placed_case_dims_mm', {}).get('length', 'N/A'),
                    "Placed Width (mm)": item_vis_data.get('placed_case_dims_mm', {}).get('width', 'N/A'),
                    "Placed Height (mm)": item_vis_data.get('placed_case_dims_mm', {}).get('height', 'N/A'),
                    "Position X (mm)": item_vis_data.get('position_mm', {}).get('x', 'N/A'),
                    "Position Y (mm)": item_vis_data.get('position_mm', {}).get('y', 'N/A'),
                    "Position Z (mm)": item_vis_data.get('position_mm', {}).get('z', 'N/A'),
                })
            report_df = pd.DataFrame(report_data)
            
            csv_export = report_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Full Pack Report (CSV)",
                data=csv_export,
                file_name="bin_packing_report.csv",
                mime="text/csv",
                key="download_report_csv"
            )
        else:
            st.caption("No packing data available to generate a report.")


        st.divider() # Divider after the statistics section

    # --- Tote Utilization Summary & Visualization (Individual Totes) ---
    st.subheader("Individual Tote Details")
    if not full_totes_summary_data:
        st.write("No totes were used or processed in the simulation (e.g., no items to pack or items too large for the configured tote).")
    else:
        if st.session_state.original_item_count > 0 :
             st.caption(f"Displaying packing results for {st.session_state.original_item_count} input items.")
        else:
            total_items_placed_for_caption = sum(len(tote.get('items',[])) for tote in full_totes_summary_data)
            if total_items_placed_for_caption > 0:
                st.caption(f"Displaying packing results for {total_items_placed_for_caption} items placed in totes.")

        # Sorting options for individual totes
        sort_container = st.container()
        with sort_container:
            scol1, scol2 = st.columns([0.6, 0.4])
            with scol1:
                sort_by_key = st.selectbox(
                    "Sort totes by:",
                    options=["Default (Tote ID)", "Utilization", "Number of Items"],
                    index=0,
                    key="tote_sort_by_selection"
                )
            with scol2:
                sort_order_asc_desc = st.radio(
                    "Order:",
                    options=["Ascending", "Descending"],
                    index=0,
                    key="tote_sort_order_selection",
                    horizontal=True
                )

        processed_totes_data = list(full_totes_summary_data)
        is_descending_sort = (sort_order_asc_desc == "Descending")

        if sort_by_key == "Utilization":
            processed_totes_data.sort(key=lambda t: t.get('utilization_percent', 0.0), reverse=is_descending_sort)
        elif sort_by_key == "Number of Items":
            processed_totes_data.sort(key=lambda t: len(t.get('items', [])), reverse=is_descending_sort)
        elif sort_by_key == "Default (Tote ID)":
            def get_tote_id_sort_key(tote_info):
                tote_id = tote_info.get('id', '')
                if isinstance(tote_id, str) and tote_id.lower().startswith("tote_"):
                    try:
                        return int(tote_id.split("_")[1])
                    except (IndexError, ValueError):
                        return str(tote_id).lower()
                elif isinstance(tote_id, (int, float)):
                    return tote_id
                return str(tote_id).lower()

            processed_totes_data.sort(key=get_tote_id_sort_key, reverse=is_descending_sort)

        for i, tote_summary_info in enumerate(processed_totes_data):
            tote_id_str = str(tote_summary_info.get('id', f'OriginalIndex_{i+1}'))
            items_in_tote = tote_summary_info.get('items', [])
            items_packed_count = len(items_in_tote)
            utilization = tote_summary_info.get('utilization_percent', 0.0)

            with st.expander(f"Tote ID: {tote_id_str} | Items: {items_packed_count} | Utilization: {utilization:.2f}%", expanded=False):

                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown("###### 3D View")
                    if items_packed_count > 0:
                        with st.spinner(f"Generating visualization for Tote {tote_id_str}..."):
                            fig = visualization.generate_tote_figure(tote_summary_info)
                            if fig:
                                st.pyplot(fig, clear_figure=True)
                                plt.close(fig)
                            else:
                                st.warning("Could not generate visualization figure for this tote.")
                    else:
                        st.caption("No items in this tote to visualize.")

                with col2:
                    st.markdown("###### Packed Item Details")
                    if items_in_tote:
                        item_details_list = []
                        for item in items_in_tote:
                            dims = item.get('chosen_orientation_dims', (None, None, None))
                            item_details_list.append({
                                "SKU": item.get('sku', 'N/A'),
                                "Length": dims[0] if dims[0] is not None else 'N/A',
                                "Width": dims[1] if dims[1] is not None else 'N/A',
                                "Height": dims[2] if dims[2] is not None else 'N/A'
                            })

                        item_details_df = pd.DataFrame(item_details_list)
                        try:
                            item_details_df['Length'] = pd.to_numeric(item_details_df['Length'], errors='coerce').round(1)
                            item_details_df['Width'] = pd.to_numeric(item_details_df['Width'], errors='coerce').round(1)
                            item_details_df['Height'] = pd.to_numeric(item_details_df['Height'], errors='coerce').round(1)
                        except Exception:
                            pass

                        st.dataframe(
                            item_details_df,
                            height=min(350, (len(item_details_list) + 1) * 35 + 3),
                            use_container_width=True
                        )
                    else:
                        st.caption("No items in this tote.")
elif not st.sidebar.button("Run Simulation", key="initial_info_trigger_dummy", disabled=True, help="This is a hidden button to ensure this block runs only if sim not run"):
    st.info("Configure parameters in the sidebar and click 'Run Simulation' to begin and see results.")
