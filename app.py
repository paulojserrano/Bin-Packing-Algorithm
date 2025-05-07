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
if 'max_rows_csv' not in st.session_state: # For CSV max rows input
    st.session_state.max_rows_csv = 0
# For progress and pause/resume
if 'simulation_running' not in st.session_state:
    st.session_state.simulation_running = False
if 'simulation_paused' not in st.session_state:
    st.session_state.simulation_paused = False
if 'simulation_progress' not in st.session_state:
    st.session_state.simulation_progress = 0.0
if 'simulation_generator' not in st.session_state:
    st.session_state.simulation_generator = None
if 'intermediate_results' not in st.session_state: # Store yielded data
    st.session_state.intermediate_results = None
if 'status_message' not in st.session_state:
    st.session_state.status_message = "Configure and run simulation."
if 'current_tote_config_for_report' not in st.session_state: # For HTML report
    st.session_state.current_tote_config_for_report = {}


# --- HTML Report Generation Function ---
def generate_styled_html_report(report_df, summary_stats_dict, tote_config):
    report_title = "Bin Packing Simulation Report"
    generation_time = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')

    html_style = """
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 0; background-color: #f4f7f6; color: #333; line-height: 1.6; }
        .report-container { max-width: 1000px; margin: 20px auto; background-color: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; text-align: center; border-bottom: 3px solid #3498db; padding-bottom: 15px; margin-bottom: 25px; font-size: 2em; }
        h2 { color: #34495e; margin-top: 35px; border-bottom: 2px solid #5dade2; padding-bottom: 8px; font-size: 1.6em;}
        .summary-table, .details-table { width: 100%; border-collapse: collapse; margin-top: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
        .summary-table th, .summary-table td, .details-table th, .details-table td {
            border: 1px solid #ddd; padding: 12px 15px; text-align: left; font-size: 0.95em;
        }
        .summary-table th { background-color: #3498db; color: white; font-weight: bold; }
        .details-table th { background-color: #5dade2; color: white; font-weight: bold; }
        .summary-table td:first-child { font-weight: bold; background-color: #ecf0f1; width: 40%; }
        .footer { text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #bdc3c7; font-size: 0.9em; color: #7f8c8d; }
        .section { margin-bottom: 35px; padding: 15px; background-color: #fdfefe; border-radius: 5px; border: 1px solid #e0e0e0;}
        .section-title { margin-bottom: 15px; }
        .config-details p, .overall-stats p { margin: 8px 0; font-size: 1em; }
        .config-details strong, .overall-stats strong { color: #2980b9; min-width: 200px; display: inline-block;}
        table { page-break-inside: auto; }
        tr { page-break-inside: avoid; page-break-after: auto; }
        thead { display: table-header-group; } /* For table header repeat on page break for printing */
        @media print {
            body { background-color: #fff; margin: 0; padding: 0;}
            .report-container { box-shadow: none; border: none; margin: 0; max-width: 100%; padding: 10px;}
            h1, h2 {color: #000;} /* Simpler colors for print */
            .summary-table th, .details-table th {background-color: #eee !important; color: #000 !important;}
            .summary-table td:first-child {background-color: #f5f5f5 !important;}
        }
    </style>
    """

    html_content = f"<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'><meta name='viewport' content='width=device-width, initial-scale=1.0'><title>{report_title}</title>{html_style}</head><body>"
    html_content += f"<div class='report-container'>"
    html_content += f"<h1>{report_title}</h1>"
    html_content += f"<p class='footer' style='margin-bottom: 20px; border-top: none;'>Report generated on: {generation_time}</p>"

    # Section: Simulation Configuration
    html_content += "<div class='section config-details'>"
    html_content += "<h2 class='section-title'>Simulation Configuration</h2>"
    html_content += f"<p><strong>Tote Length:</strong> {tote_config.get('TOTE_MAX_LENGTH', 'N/A')} mm</p>"
    html_content += f"<p><strong>Tote Width:</strong> {tote_config.get('TOTE_MAX_WIDTH', 'N/A')} mm</p>"
    html_content += f"<p><strong>Tote Height:</strong> {tote_config.get('TOTE_MAX_HEIGHT', 'N/A')} mm</p>"
    html_content += f"<p><strong>Height Map Resolution:</strong> {tote_config.get('HEIGHT_MAP_RESOLUTION', 'N/A')} mm</p>"
    html_content += "</div>"

    # Section: Overall Statistics
    html_content += "<div class='section overall-stats'>"
    html_content += "<h2 class='section-title'>Overall Packing Statistics</h2>"
    html_content += f"<p><strong>Total Totes Used:</strong> {summary_stats_dict.get('total_totes_used', 'N/A')}</p>"
    html_content += f"<p><strong>Total Cases Placed:</strong> {summary_stats_dict.get('total_items_placed', 'N/A')}</p>"
    html_content += f"<p><strong>Cases That Did Not Fit:</strong> {summary_stats_dict.get('unplaced_items_count', 'N/A')}</p>"
    avg_util_str = f"{summary_stats_dict.get('average_utilization', 0.0):.2f}%" if summary_stats_dict.get('average_utilization') is not None else "N/A"
    min_util_str = f"{summary_stats_dict.get('min_utilization', 0.0):.2f}%" if summary_stats_dict.get('min_utilization') is not None else "N/A"
    median_util_str = f"{summary_stats_dict.get('median_utilization', 0.0):.2f}%" if summary_stats_dict.get('median_utilization') is not None else "N/A"
    max_util_str = f"{summary_stats_dict.get('max_utilization', 0.0):.2f}%" if summary_stats_dict.get('max_utilization') is not None else "N/A"
    html_content += f"<p><strong>Average Tote Utilization:</strong> {avg_util_str}</p>"
    html_content += f"<p><strong>Minimum Tote Utilization:</strong> {min_util_str}</p>"
    html_content += f"<p><strong>Median Tote Utilization:</strong> {median_util_str}</p>"
    html_content += f"<p><strong>Maximum Tote Utilization:</strong> {max_util_str}</p>"
    html_content += "</div>"

    # Section: Detailed Packing Report
    html_content += "<div class='section'>"
    html_content += "<h2 class='section-title'>Detailed Packing List per Tote</h2>"
    if not report_df.empty:
        # Group by Tote ID for better readability if multiple totes
        if 'Tote ID' in report_df.columns:
            for tote_id, group_df in report_df.groupby('Tote ID'):
                html_content += f"<h3>Tote: {tote_id}</h3>"
                html_content += group_df.to_html(index=False, escape=False, classes="details-table", border=0)
                html_content += "<br>" # Add some space between tote tables
        else: # Fallback if no Tote ID column (should not happen with current report_df)
            html_content += report_df.to_html(index=False, escape=False, classes="details-table", border=0)
    else:
        html_content += "<p>No items were packed in this simulation run.</p>"
    html_content += "</div>"

    html_content += f"<div class='footer'>End of Report</div>"
    html_content += "</div></body></html>"
    return html_content

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

    if uploaded_file is not None: # Only show if a file is uploaded
        st.sidebar.number_input(
            "Max rows to load from CSV (0 for all)",
            min_value=0,
            value=st.session_state.get('max_rows_csv', 0), 
            step=100,
            key="max_rows_csv", # Links to st.session_state.max_rows_csv
            help="Enter 0 or leave blank to load all rows. Headers are always read."
        )

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


# --- Run/Pause Simulation Buttons ---
run_col, pause_col = st.sidebar.columns(2)

# Run Button
if run_col.button("Run Simulation", key="run_button", type="primary", disabled=st.session_state.get('simulation_running', False)):
    # Reset all relevant states
    st.session_state.simulation_ran = False
    st.session_state.simulation_running = True # Start the process
    st.session_state.simulation_paused = False
    st.session_state.simulation_progress = 0.0
    st.session_state.simulation_generator = None
    st.session_state.simulation_results = {'visualization_output_list': [], 'full_totes_summary_data': []}
    st.session_state.intermediate_results = None
    st.session_state.original_item_count = 0 # Reset item count
    st.session_state.status_message = "Initializing simulation..."

    dynamic_tote_config = {
        "TOTE_MAX_LENGTH": int(tote_length_input),
        "TOTE_MAX_WIDTH": int(tote_width_input),
        "TOTE_MAX_HEIGHT": int(tote_height_input),
        "TOTE_MAX_VOLUME": int(tote_length_input * tote_width_input * tote_height_input),
        "HEIGHT_MAP_RESOLUTION": int(height_map_resolution_input),
        "GRID_DIM_X": max(1, math.ceil(int(tote_length_input) / int(height_map_resolution_input))),
        "GRID_DIM_Y": max(1, math.ceil(int(tote_width_input) / int(height_map_resolution_input)))
    }

    # --- Prepare simulation ---
    simulation_can_proceed = False
    current_input_cases = []
    st.session_state.current_tote_config_for_report = dynamic_tote_config # Save for report

    if case_data_source == "Generate Random Cases":
        # Generate cases directly
        current_input_cases = simulation.generate_test_cases(
            num_cases=int(num_random_cases_input),
            seed=int(random_seed_input),
            current_tote_config=dynamic_tote_config
        )
        if current_input_cases:
            st.session_state.original_item_count = len(current_input_cases)
            simulation_can_proceed = True
            st.session_state.status_message = f"Generated {len(current_input_cases)} random cases. Starting simulation..."
        else:
            st.warning("No random cases were generated.")
            st.session_state.simulation_running = False # Stop if no cases

    elif case_data_source == "Upload CSV File":
        if uploaded_file is not None:
            len_col = st.session_state.column_mappings.get('length_col')
            wid_col = st.session_state.column_mappings.get('width_col')
            hei_col = st.session_state.column_mappings.get('height_col')
            sku_col_map = st.session_state.column_mappings.get('sku_col')

            if not all([len_col, wid_col, hei_col]):
                st.error("Column mapping incomplete. Please select columns for Length, Width, and Height.")
                st.session_state.simulation_running = False # Stop
            else:
                try:
                    uploaded_file.seek(0)
                    columns_to_read = list(set(filter(None, [len_col, wid_col, hei_col, sku_col_map if sku_col_map != "Auto-generate SKU" else None])))
                    if not all(c in st.session_state.csv_headers for c in [len_col, wid_col, hei_col]):
                         st.error("Essential mapped columns not found in CSV headers.")
                         st.session_state.simulation_running = False # Stop
                    else:
                        max_rows_val = st.session_state.get('max_rows_csv', 0)
                        nrows_param = int(max_rows_val) if max_rows_val > 0 else None
                        df = pd.read_csv(uploaded_file, usecols=columns_to_read, nrows=nrows_param)

                        # Basic validation (can be expanded)
                        if not all(pd.api.types.is_numeric_dtype(df[col]) for col in [len_col, wid_col, hei_col]):
                            st.error("Length, Width, Height columns must be numeric.")
                            st.session_state.simulation_running = False # Stop
                        elif not all((df[col] > 0).all() for col in [len_col, wid_col, hei_col]):
                             st.error("Length, Width, Height values must be positive.")
                             st.session_state.simulation_running = False # Stop
                        else:
                            for index, row in df.iterrows():
                                sku_val = f"CSV_SKU_{index+1}"
                                if sku_col_map and sku_col_map != "Auto-generate SKU" and sku_col_map in df.columns:
                                    sku_val = str(row[sku_col_map])
                                current_input_cases.append({
                                    "sku": sku_val, "length": float(row[len_col]),
                                    "width": float(row[wid_col]), "height": float(row[hei_col])
                                })
                            st.session_state.original_item_count = len(current_input_cases)
                            simulation_can_proceed = True
                            st.session_state.status_message = f"Loaded {len(current_input_cases)} cases from CSV. Starting simulation..."
                except Exception as e:
                    st.error(f"Error processing CSV: {e}")
                    st.session_state.simulation_running = False # Stop
        else:
            st.error("Please upload a CSV file.")
            st.session_state.simulation_running = False # Stop

    # --- Initialize Generator if ready ---
    if simulation_can_proceed and current_input_cases:
        st.session_state.simulation_generator = simulation.run_simulation_for_visualization_data(
            case_data_list=current_input_cases,
            current_tote_config=dynamic_tote_config
        )
        st.rerun() # Start consuming the generator in the main script body
    elif not simulation_can_proceed:
        st.session_state.simulation_running = False # Ensure it's stopped if setup failed
        st.rerun()

# Pause/Resume Button
pause_label = "Pause" if not st.session_state.get('simulation_paused') else "Resume"
if pause_col.button(pause_label, key="pause_button", disabled=not st.session_state.get('simulation_running', False)):
    st.session_state.simulation_paused = not st.session_state.simulation_paused
    st.session_state.status_message = "Simulation Paused." if st.session_state.simulation_paused else "Resuming simulation..."
    st.rerun()


# --- Simulation Progress Display Area ---
st.header("Simulation Process & Results")
progress_area = st.empty() # Placeholder for progress bar and status

# --- Consume Generator (if running and not paused) ---
if st.session_state.get('simulation_running') and not st.session_state.get('simulation_paused'):
    if st.session_state.simulation_generator:
        try:
            # Get the next update from the simulation generator
            yielded_data = next(st.session_state.simulation_generator, None)

            if yielded_data:
                st.session_state.simulation_progress = yielded_data.get("progress", 0.0)
                st.session_state.status_message = yielded_data.get("status_message", "Processing...")
                st.session_state.intermediate_results = yielded_data # Store the whole dict

                # Check if this is the final yield
                if yielded_data.get("is_final", False):
                    st.session_state.simulation_running = False
                    st.session_state.simulation_paused = False
                    st.session_state.simulation_ran = True # Mark as completed
                    # Store final results properly
                    st.session_state.simulation_results['visualization_output_list'] = yielded_data.get('intermediate_vis_data', [])
                    st.session_state.simulation_results['full_totes_summary_data'] = yielded_data.get('intermediate_totes_data', [])
                    st.session_state.simulation_generator = None # Clear generator
                    st.success("Simulation finished!") # Show success message
                    st.session_state.status_message = yielded_data.get("status_message", "Simulation Complete.")
                st.rerun() # Rerun to process next step or update UI after final step
            else:
                # Generator exhausted unexpectedly or finished without final flag (handle gracefully)
                st.session_state.simulation_running = False
                st.session_state.simulation_paused = False
                st.session_state.simulation_ran = True # Assume finished if generator ends
                st.session_state.simulation_generator = None
                # Use the last known intermediate results as final if necessary
                if st.session_state.intermediate_results:
                     st.session_state.simulation_results['visualization_output_list'] = st.session_state.intermediate_results.get('intermediate_vis_data', [])
                     st.session_state.simulation_results['full_totes_summary_data'] = st.session_state.intermediate_results.get('intermediate_totes_data', [])
                     st.session_state.status_message = st.session_state.intermediate_results.get("status_message", "Simulation Ended.")
                else:
                     st.session_state.status_message = "Simulation ended unexpectedly."
                st.rerun()

        except Exception as e:
            st.error(f"An error occurred during simulation: {e}")
            st.session_state.simulation_running = False
            st.session_state.simulation_paused = False
            st.session_state.simulation_generator = None
            st.rerun()

# --- Update Progress Area ---
with progress_area.container():
    if st.session_state.get('simulation_running') or st.session_state.get('simulation_ran'):
        progress_value = st.session_state.get('simulation_progress', 0.0)
        status_text = st.session_state.get('status_message', '')
        st.progress(progress_value, text=f"{status_text} ({progress_value:.0%})")
    else:
        st.info(st.session_state.get('status_message', "Configure parameters and click 'Run Simulation'."))


# --- Display Results (Show intermediate if paused, final if ran) ---
results_to_display = None
if st.session_state.get('simulation_paused') and st.session_state.get('intermediate_results'):
    # Show intermediate results when paused
    results_to_display = st.session_state.intermediate_results
    st.warning("Simulation Paused. Showing intermediate results.")
elif st.session_state.get('simulation_ran'):
    # Show final results after completion
    results_to_display = {
        'intermediate_totes_data': st.session_state.simulation_results['full_totes_summary_data'],
        'intermediate_vis_data': st.session_state.simulation_results['visualization_output_list']
        # We might need to reconstruct the unplaceable log if needed from final results, or store it separately
    }

if results_to_display:
    full_totes_summary_data = results_to_display.get('intermediate_totes_data', [])
    visualization_output_list = results_to_display.get('intermediate_vis_data', [])
    # unplaceable_log = results_to_display.get('unplaceable_log', []) # If needed


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
            for item_vis_data in visualization_output_list: # This is the detailed per-item-placement list
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

            # Prepare data for styled HTML report
            current_tote_config = st.session_state.get('current_tote_config_for_report', {})
            report_summary_stats = {
                "total_totes_used": total_totes_used,
                "total_items_placed": total_items_placed_in_stats,
                "unplaced_items_count": unplaced_items_count,
                "average_utilization": statistics.mean(utilization_percentages) if utilization_percentages else None,
                "min_utilization": min(utilization_percentages) if utilization_percentages else None,
                "median_utilization": statistics.median(utilization_percentages) if utilization_percentages else None,
                "max_utilization": max(utilization_percentages) if utilization_percentages else None,
            }
            
            csv_export = report_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Full Pack Report (CSV)",
                data=csv_export,
                file_name="bin_packing_report.csv",
                mime="text/csv",
                key="download_report_csv"
            )

            # Generate styled HTML
            html_content_styled = generate_styled_html_report(report_df, report_summary_stats, current_tote_config)
            html_export_styled = html_content_styled.encode('utf-8')
            st.download_button(
                label="Download Full Pack Report (HTML)",
                data=html_export_styled,
                file_name="bin_packing_report_styled.html",
                mime="text/html",
                key="download_report_html_styled"
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

                col1, col2 = st.columns([2, 1]) # Keep existing layout for details and plot

                with col1:
                    st.markdown("###### 3D View")
                    if items_packed_count > 0:
                        # Use a unique key for the button based on tote_id_str
                        button_key = f"load_plot_button_{tote_id_str}_{i}" # Add index i for more uniqueness if tote_id_str could repeat (e.g. "Tote_1" from two runs)
                        
                        # Check if plot was loaded in this session for this tote
                        plot_loaded_session_key = f"plot_loaded_{tote_id_str}_{i}"

                        if st.button("Load/Refresh 3D View", key=button_key):
                            with st.spinner(f"Generating visualization for Tote {tote_id_str}..."):
                                fig = visualization.generate_tote_figure(tote_summary_info)
                                if fig:
                                    # Store figure in session state to persist it if button is not pressed again
                                    # but this can consume memory. For now, direct display.
                                    st.session_state[plot_loaded_session_key] = fig 
                                    # No, pyplot will display it. If we want to cache, need to handle display from cache.
                                    # Simpler: just display it. It will disappear if expander re-renders without button press.
                                    st.pyplot(fig, clear_figure=True)
                                    plt.close(fig) # Close the figure to free memory after displaying
                                    # To make it "stick" better, one might store the fig in session_state
                                    # and then check session_state to display it, but that's more complex.
                                    # This implementation: click to load. It shows. If page reloads/expander closes & opens, click again.
                                else:
                                    st.warning("Could not generate visualization figure for this tote.")
                                    if plot_loaded_session_key in st.session_state:
                                        del st.session_state[plot_loaded_session_key] # Clear if failed
                        # else:
                            # This part would be for displaying a cached plot if we implemented caching.
                            # For now, plot only shows on button click.
                            # if plot_loaded_session_key in st.session_state and st.session_state[plot_loaded_session_key]:
                            #    st.pyplot(st.session_state[plot_loaded_session_key], clear_figure=True)


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
# The initial info message is now handled by the progress_area logic when not running/ran.
