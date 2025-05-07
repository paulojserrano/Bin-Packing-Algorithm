# app.py
import streamlit as st
import pandas as pd
import math # For calculations
import matplotlib.pyplot as plt # Added for plt.close() and histogram
import statistics # For median
import numpy as np # For histogram binning and other calculations if needed

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
    "Tote Length", min_value=50, value=config.TOTE_MAX_LENGTH, step=10, key="tote_length"
)
tote_width_input = st.sidebar.number_input(
    "Tote Width", min_value=50, value=config.TOTE_MAX_WIDTH, step=10, key="tote_width"
)
tote_height_input = st.sidebar.number_input(
    "Tote Height", min_value=50, value=config.TOTE_MAX_HEIGHT, step=10, key="tote_height"
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
        # st.session_state.case_csv_uploader = None # May be needed if uploader value persists

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
                df_headers = pd.read_csv(uploaded_file, nrows=0)
                st.session_state.csv_headers = df_headers.columns.tolist()
                st.session_state.column_mappings = {k: None for k in st.session_state.column_mappings}
                uploaded_file.seek(0) 
                st.sidebar.success(f"File '{uploaded_file.name}' loaded. Map columns below.")
            except Exception as e:
                st.sidebar.error(f"Error reading CSV headers: {e}")
                st.session_state.csv_headers = [] 
                st.session_state.uploaded_file_name = None 
    else:
        if st.session_state.uploaded_file_name is not None:
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
    elif uploaded_file and not st.session_state.csv_headers: 
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
                    uploaded_file.seek(0) 
                    df = pd.read_csv(uploaded_file)
                    
                    required_mapped_cols = {'Length': len_col, 'Width': wid_col, 'Height': hei_col}
                    valid_data = True
                    for std_name, actual_col in required_mapped_cols.items():
                        if actual_col not in df.columns:
                            st.error(f"Mapped column '{actual_col}' for {std_name} not found in CSV. Please check mappings or CSV file.")
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
                            if sku_col_map and sku_col_map != "Auto-generate SKU":
                                if sku_col_map in df.columns: 
                                    sku_val = str(row[sku_col_map]) 
                                else: 
                                    st.warning(f"SKU column '{sku_col_map}' selected in mapping but not found in CSV. Using auto-generated SKU for row {index+1}.")
                            
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
        # We don't need main_results_area.empty() anymore as content will be drawn sequentially
        st.header("Simulation Process & Results") # This header will appear once
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
        pass 
    elif not simulation_can_proceed and current_input_cases: 
        st.error("Simulation cannot proceed despite having case data. Check error messages above in the main panel or sidebar.")


# --- Display Results (Statistics first, then individual totes) ---
if st.session_state.simulation_ran:
    full_totes_summary_data = st.session_state.simulation_results['full_totes_summary_data']

    # --- Overall Statistics Section (Moved Up) ---
    if full_totes_summary_data: # Only show stats if there's data
        st.subheader("Overall Tote Utilization Statistics")
        utilization_percentages = [tote.get('utilization_percent', 0.0) for tote in full_totes_summary_data if tote.get('utilization_percent') is not None]
        
        if utilization_percentages:
            avg_util = statistics.mean(utilization_percentages)
            min_util = min(utilization_percentages)
            max_util = max(utilization_percentages) 
            median_util = statistics.median(utilization_percentages)
            
            stats_col1, stats_col2 = st.columns(2)
            with stats_col1:
                st.metric(label="Average Tote Utilization", value=f"{avg_util:.2f}%")
                st.metric(label="Minimum Tote Utilization", value=f"{min_util:.2f}%")
            with stats_col2:
                st.metric(label="Maximum Tote Utilization", value=f"{max_util:.2f}%") 
                st.metric(label="Median Tote Utilization", value=f"{median_util:.2f}%")

            # Histogram of Tote Utilization
            st.markdown("###### Distribution of Tote Utilizations")
            # Further reduce figsize and adjust fonts for a much smaller histogram
            fig_hist, ax_hist = plt.subplots(figsize=(3.0, 1.75)) # Significantly smaller
            ax_hist.hist(utilization_percentages, bins='auto', color='skyblue', edgecolor='black')
            ax_hist.set_title('Tote Utilization %', fontsize=7) # Shorter title, smaller font
            ax_hist.set_xlabel('Utilization %', fontsize=6) 
            ax_hist.set_ylabel('No. of Totes', fontsize=6) # Shorter label
            ax_hist.tick_params(axis='both', which='major', labelsize=5) 
            fig_hist.tight_layout(pad=0.3) # Adjust padding
            st.pyplot(fig_hist, use_container_width=False) # IMPORTANT: use_container_width=False
            plt.close(fig_hist) # Close the histogram figure

        else:
            st.write("No utilization data available to calculate statistics or plot histogram.")
        st.divider() # Divider after stats

    # --- Tote Utilization Summary & Visualization (Individual Totes) ---
    st.subheader("Individual Tote Details") 
    if not full_totes_summary_data:
        st.write("No totes were used or processed in the simulation (e.g., no items to pack or items too large for the configured tote).")
    else:
        if st.session_state.original_item_count > 0 :
             st.caption(f"Displaying packing results for {st.session_state.original_item_count} input items.")
        else: 
            total_items_placed = sum(len(tote.get('items',[])) for tote in full_totes_summary_data)
            if total_items_placed > 0:
                st.caption(f"Displaying packing results for {total_items_placed} items placed in totes.")

        for i, tote_summary_info in enumerate(full_totes_summary_data):
            tote_id_str = str(tote_summary_info.get('id', f'Tote_{i+1}')) 
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

