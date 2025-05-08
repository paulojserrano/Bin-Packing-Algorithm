# app.py
import streamlit as st
import pandas as pd
import math # For calculations
import matplotlib.pyplot as plt # Added for plt.close() and histogram
import statistics # For median
import numpy as np # For histogram binning and other calculations if needed
from collections import Counter, defaultdict # For new statistics
import io # Added for Task 2
import base64 # Added for Task 2

# Import existing modules from the project
import simulation
import core_utils
import visualization # Now includes generate_tote_figure
import config # To get default values initially

# --- UI Configuration ---
st.set_page_config(layout="wide", page_title="Bin Packing Simulation")
st.title("Interactive Bin Packing Simulation")

# --- Algorithm Explanation (Collapsible) ---
METHODOLOGY_TEXT = """
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
    """
with st.expander("Algorithm Overview and Methodology", expanded=False):
    st.markdown(METHODOLOGY_TEXT)
st.divider() # Adds a visual separator

# --- Initialize Session State ---
# For CSV column mapping
if 'csv_sampling_method' not in st.session_state: # Added for CSV random/sequential
    st.session_state.csv_sampling_method = 'Sequential'
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
        'full_totes_summary_data': [],
        'unplaceable_items_log': [] # Added for unplaced items
    }
if 'original_item_count' not in st.session_state: # To store original item count for captions
    st.session_state.original_item_count = 0
if 'max_rows_csv' not in st.session_state: # For CSV max rows input
    st.session_state.max_rows_csv = 0
if 'csv_random_seed' not in st.session_state: # Added for CSV random sampling seed
    st.session_state.csv_random_seed = 42
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
def generate_styled_html_report(report_df, summary_stats_dict, tote_config, all_totes_data, unplaceable_items_log):
    report_title = "Bin Packing Simulation Report"
    generation_time = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')

    # --- Methodology Section ---
    methodology_section_html = "<div class='section methodology-section'>"
    methodology_section_html += "<h2 class='section-title'>Algorithm Overview and Methodology</h2>"
    methodology_html = METHODOLOGY_TEXT.replace("**", "<strong>") # Basic replacement for bold
    # Consider a proper Markdown to HTML converter for more complex markdown in METHODOLOGY_TEXT
    methodology_section_html += f"<div style='font-size: 0.9em; text-align: left; white-space: pre-wrap;'>{methodology_html}</div>"
    methodology_section_html += "</div>"

    # --- Simulation Configuration Section ---
    sim_config_html = "<div class='section config-details'>"
    sim_config_html += "<h2 class='section-title'>Simulation Configuration</h2>"
    sim_config_html += f"<p><strong>Tote Length:</strong> {tote_config.get('TOTE_MAX_LENGTH', 'N/A')} mm</p>"
    sim_config_html += f"<p><strong>Tote Width:</strong> {tote_config.get('TOTE_MAX_WIDTH', 'N/A')} mm</p>"
    sim_config_html += f"<p><strong>Tote Height:</strong> {tote_config.get('TOTE_MAX_HEIGHT', 'N/A')} mm</p>"
    sim_config_html += f"<p><strong>Height Map Resolution:</strong> {tote_config.get('HEIGHT_MAP_RESOLUTION', 'N/A')} mm</p>"
    sim_config_html += "</div>"

    # --- Overall Packing Statistics Section ---
    overall_stats_html = "<div class='section overall-stats'>"
    overall_stats_html += "<h2 class='section-title'>Overall Packing Statistics</h2>"
    overall_stats_html += f"<p><strong>Total Totes Used:</strong> {summary_stats_dict.get('total_totes_used', 'N/A')}</p>"
    overall_stats_html += f"<p><strong>Total Cases Placed:</strong> {summary_stats_dict.get('total_items_placed', 'N/A')}</p>"
    overall_stats_html += f"<p><strong>Cases That Did Not Fit:</strong> {summary_stats_dict.get('unplaced_items_count', 'N/A')}</p>"
    avg_util_str = f"{summary_stats_dict.get('average_utilization', 0.0):.2f}%" if summary_stats_dict.get('average_utilization') is not None else "N/A"
    min_util_str = f"{summary_stats_dict.get('min_utilization', 0.0):.2f}%" if summary_stats_dict.get('min_utilization') is not None else "N/A"
    median_util_str = f"{summary_stats_dict.get('median_utilization', 0.0):.2f}%" if summary_stats_dict.get('median_utilization') is not None else "N/A"
    max_util_str = f"{summary_stats_dict.get('max_utilization', 0.0):.2f}%" if summary_stats_dict.get('max_utilization') is not None else "N/A"
    overall_stats_html += f"<p><strong>Average Tote Utilization:</strong> {avg_util_str}</p>"
    overall_stats_html += f"<p><strong>Minimum Tote Utilization:</strong> {min_util_str}</p>"
    overall_stats_html += f"<p><strong>Median Tote Utilization:</strong> {median_util_str}</p>"
    overall_stats_html += f"<p><strong>Maximum Tote Utilization:</strong> {max_util_str}</p>"
    avg_items_str = f"{summary_stats_dict.get('avg_items_per_tote', 0.0):.2f}"
    overall_stats_html += f"<p><strong>Average Items per Tote:</strong> {avg_items_str}</p>"
    single_case_totes_count_str = str(summary_stats_dict.get('totes_with_one_item', 'N/A'))
    overall_stats_html += f"<p><strong>Single-Case Totes (Count):</strong> {single_case_totes_count_str}</p>"
    single_case_totes_perc_str = f"{summary_stats_dict.get('percentage_single_case_totes', 0.0):.2f}%"
    overall_stats_html += f"<p><strong>Single-Case Totes (%):</strong> {single_case_totes_perc_str}</p>"
    unique_skus_str = str(summary_stats_dict.get('num_unique_skus_placed', 'N/A'))
    overall_stats_html += f"<p><strong>Total Unique SKUs Placed:</strong> {unique_skus_str}</p>"
    avg_totes_sku_str = f"{summary_stats_dict.get('avg_totes_per_sku', 0.0):.2f}"
    overall_stats_html += f"<p><strong>Avg. Totes per SKU:</strong> {avg_totes_sku_str}</p>"
    most_freq_sku_str = str(summary_stats_dict.get('most_frequent_sku', 'N/A'))
    overall_stats_html += f"<p><strong>Most Frequent SKU:</strong> {most_freq_sku_str}</p>"
    utilization_percentages_for_hist = [
        tote.get('utilization_percent', 0.0)
        for tote in all_totes_data
        if tote.get('utilization_percent') is not None
    ]
    if utilization_percentages_for_hist:
        overall_stats_html += "<h4 style='margin-top: 25px; margin-bottom: 10px; color: #495057;'>Distribution of Tote Utilizations:</h4>"
        try:
            fig_hist_report, ax_hist_report = plt.subplots(figsize=(6, 3.5))
            ax_hist_report.hist(utilization_percentages_for_hist, bins='auto', color='cornflowerblue', edgecolor='black', alpha=0.75)
            ax_hist_report.set_title('Tote Utilization Distribution', fontsize=11)
            ax_hist_report.set_xlabel('Utilization %', fontsize=9)
            ax_hist_report.set_ylabel('Number of Totes', fontsize=9)
            ax_hist_report.tick_params(axis='both', which='major', labelsize=8)
            ax_hist_report.grid(axis='y', linestyle='--', alpha=0.7)
            fig_hist_report.tight_layout(pad=0.8)
            buf_hist = io.BytesIO()
            fig_hist_report.savefig(buf_hist, format='png', dpi=90)
            buf_hist.seek(0)
            img_base64_hist = base64.b64encode(buf_hist.getvalue()).decode('utf-8')
            overall_stats_html += f"<div class='tote-image-container' style='max-width: 550px; margin-left: auto; margin-right: auto;'><img src='data:image/png;base64,{img_base64_hist}' alt='Tote Utilization Histogram'></div>"
            plt.close(fig_hist_report)
        except Exception as e:
            overall_stats_html += f"<p><em>Error generating utilization histogram: {str(e)}</em></p>"
            if 'fig_hist_report' in locals() and fig_hist_report: plt.close(fig_hist_report)
    else:
        overall_stats_html += "<p><em>No utilization data available for histogram.</em></p>"
    overall_stats_html += "</div>"

    # --- Unplaced Items Section ---
    unplaced_items_html = ""
    if unplaceable_items_log:
        unplaced_items_html += "<div class='section'>"
        unplaced_items_html += "<details class='tote-details-collapsible' style='border-color: #ffc107;' open>"
        unplaced_items_html += "<summary style='background-color: #fff3cd; color: #856404; font-weight: bold;'>"
        unplaced_items_html += f"Unplaced Items ({len(unplaceable_items_log)}) - Click to Expand/Collapse"
        unplaced_items_html += "</summary>"
        unplaced_items_html += "<div class='tote-content'>"
        unplaced_items_html += "<h4>Items that could not be packed:</h4>"
        unplaced_df = pd.DataFrame(unplaceable_items_log)
        if not unplaced_df.empty and 'dimensions' in unplaced_df.columns:
            valid_dimensions_series = unplaced_df['dimensions'].dropna()
            if not valid_dimensions_series.empty:
                first_dim_element = valid_dimensions_series.iloc[0]
                if isinstance(first_dim_element, (tuple, list)) and len(first_dim_element) == 3:
                    def extract_dims(dim_data):
                        if isinstance(dim_data, (tuple, list)) and len(dim_data) == 3: return dim_data
                        return (None, None, None)
                    dims_list = unplaced_df['dimensions'].apply(extract_dims).tolist()
                    dims_as_df = pd.DataFrame(dims_list, index=unplaced_df.index, columns=['Original Length', 'Original Width', 'Original Height'])
                    unplaced_df = pd.concat([unplaced_df.drop(columns=['dimensions']), dims_as_df], axis=1)
                elif isinstance(first_dim_element, dict):
                    dims_from_dict_df = unplaced_df['dimensions'].apply(pd.Series)
                    rename_map_dict = {}
                    if 'length' in dims_from_dict_df.columns: rename_map_dict['length'] = 'Original Length'
                    if 'width' in dims_from_dict_df.columns: rename_map_dict['width'] = 'Original Width'
                    if 'height' in dims_from_dict_df.columns: rename_map_dict['height'] = 'Original Height'
                    if rename_map_dict: dims_from_dict_df = dims_from_dict_df.rename(columns=rename_map_dict)
                    unplaced_df = pd.concat([unplaced_df.drop(columns=['dimensions']), dims_from_dict_df], axis=1)
        display_cols_unplaced = {}
        if 'sku' in unplaced_df.columns: display_cols_unplaced['sku'] = 'SKU'
        if 'reason' in unplaced_df.columns: display_cols_unplaced['reason'] = 'Reason'
        if 'Original Length' in unplaced_df.columns: display_cols_unplaced['Original Length'] = 'Length (mm)'
        if 'Original Width' in unplaced_df.columns: display_cols_unplaced['Original Width'] = 'Width (mm)'
        if 'Original Height' in unplaced_df.columns: display_cols_unplaced['Original Height'] = 'Height (mm)'
        if display_cols_unplaced and not unplaced_df.empty:
            cols_to_select_for_display = [col for col in display_cols_unplaced.keys() if col in unplaced_df.columns]
            if cols_to_select_for_display:
                unplaced_display_df = unplaced_df[cols_to_select_for_display].rename(columns=display_cols_unplaced)
                standard_order = ['SKU', 'Reason', 'Length (mm)', 'Width (mm)', 'Height (mm)']
                ordered_cols_present = [col for col in standard_order if col in unplaced_display_df.columns]
                remaining_cols = [col for col in unplaced_display_df.columns if col not in ordered_cols_present]
                final_ordered_cols = ordered_cols_present + remaining_cols
                unplaced_display_df = unplaced_display_df[final_ordered_cols]
                for col_name in ['Length (mm)', 'Width (mm)', 'Height (mm)']:
                    if col_name in unplaced_display_df.columns:
                        unplaced_display_df[col_name] = pd.to_numeric(unplaced_display_df[col_name], errors='coerce').apply(lambda x: f"{x:.1f}" if pd.notnull(x) else "N/A")
                unplaced_items_html += unplaced_display_df.to_html(index=False, escape=False, classes="details-table", border=0)
            else: unplaced_items_html += "<p>No displayable data found for unplaced items (columns missing or not mapped).</p>"
        elif unplaced_df.empty: unplaced_items_html += "<p>No unplaced items to display.</p>"
        else:
            unplaced_items_html += "<p>Unplaced item data is not in the expected format or no columns to display. Raw data:</p>"
            unplaced_items_html += "<pre>" + str(unplaceable_items_log) + "</pre>"
        unplaced_items_html += "</div></details></div>"
    else:
        unplaced_items_html += "<div class='section'><p>All items were successfully placed, or no items were attempted for packing.</p></div>"

    # --- Individual Tote Details Controls ---
    tote_details_controls_html = """
    <div class="sort-controls" style="margin-bottom: 20px; padding: 10px; background-color: #f9f9f9; border-radius: 5px; border: 1px solid #eee; display: flex; align-items: center; flex-wrap: wrap;">
        <label for="sort-key" style="margin-right: 5px; font-weight: 500;">Sort by:</label>
        <select id="sort-key" style="margin-right: 15px; padding: 6px 8px; border-radius: 4px; border: 1px solid #ccc; font-size: 0.95em;">
            <option value="id">Tote ID</option>
            <option value="items">Number of Items</option>
            <option value="utilization">Utilization</option>
        </select>
        <label for="sort-order" style="margin-right: 5px; font-weight: 500;">Order:</label>
        <select id="sort-order" style="margin-right: 15px; padding: 6px 8px; border-radius: 4px; border: 1px solid #ccc; font-size: 0.95em;">
            <option value="asc">Ascending</option>
            <option value="desc">Descending</option>
        </select>
        <button onclick="sortToteDetailsHTMLReport()" style="padding: 6px 12px; border-radius: 4px; border: none; background-color: #007bff; color: white; cursor: pointer; font-size: 0.95em; font-weight: 500;">Sort</button>
        <label for="skuFilterInput" style="margin-left: 20px; margin-right: 5px; font-weight: 500;">Filter by SKU:</label>
        <input type="text" id="skuFilterInput" placeholder="Enter SKU..." style="padding: 6px 8px; border-radius: 4px; border: 1px solid #ccc; font-size: 0.95em; margin-right: 5px;">
        <button onclick="filterTotesBySKU()" style="padding: 6px 12px; border-radius: 4px; border: none; background-color: #28a745; color: white; cursor: pointer; font-size: 0.95em; font-weight: 500;">Filter</button>
    </div>
    """

    # --- Individual Tote Details Container ---
    tote_details_container_html = ""
    if all_totes_data:
        for tote_summary_info in all_totes_data:
            tote_id = tote_summary_info.get('id', 'Unknown Tote')
            items_in_tote_list = tote_summary_info.get('items', [])
            utilization = tote_summary_info.get('utilization_percent', 0.0)
            items_packed_count = len(items_in_tote_list)
            skus_in_tote = ",".join(set(item.get('sku', '') for item in items_in_tote_list if item.get('sku')))
            tote_id_attr = str(tote_id)
            items_packed_count_attr = str(items_packed_count)
            utilization_attr = f"{utilization:.2f}"
            tote_details_container_html += f"<div class='tote-wrapper' style='margin-bottom: 12px;'>"
            tote_details_container_html += f"<details class='tote-details-collapsible' data-tote-id='{tote_id_attr}' data-items='{items_packed_count_attr}' data-utilization='{utilization_attr}' data-skus='{skus_in_tote}'>"
            tote_details_container_html += f"<summary><strong>Tote ID: {tote_id}</strong> | Items: {items_packed_count} | Utilization: {utilization:.2f}%</summary>"
            tote_details_container_html += "<div class='tote-content'>"
            fig = visualization.generate_tote_figure(tote_summary_info)
            if fig:
                try:
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', bbox_inches='tight', dpi=75)
                    buf.seek(0)
                    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                    tote_details_container_html += f"<div class='tote-image-container'><img src='data:image/png;base64,{img_base64}' alt='Tote {tote_id} visualization'></div>"
                    plt.close(fig)
                except Exception as e:
                    tote_details_container_html += f"<p><em>Error generating image for Tote {tote_id}: {str(e)}</em></p>"
                    if fig: plt.close(fig)
            else:
                tote_details_container_html += f"<p><em>Could not generate visualization for Tote {tote_id}.</em></p>"
            tote_items_df = report_df[report_df['Tote ID'] == tote_id]
            if not tote_items_df.empty:
                tote_details_container_html += "<h4>Packed Items:</h4>"
                tote_details_container_html += tote_items_df.to_html(index=False, escape=False, classes="details-table", border=0)
            else:
                tote_details_container_html += "<p>No item details found for this tote in the main report.</p>"
            tote_details_container_html += "</div></details></div>"
    else:
        tote_details_container_html += "<p>No tote data available for detailed view.</p>"

    # --- Report Scripts ---
    report_scripts_html = """
    <script>
    function sortToteDetailsHTMLReport() {
        const container = document.getElementById('tote-details-container');
        if (!container) { console.error("Tote details container not found."); return; }
        const sortKey = document.getElementById('sort-key').value;
        const sortOrder = document.getElementById('sort-order').value;
        const toteWrappers = Array.from(container.getElementsByClassName('tote-wrapper'));
        toteWrappers.sort((a, b) => {
            const detailsA = a.querySelector('details');
            const detailsB = b.querySelector('details');
            if (!detailsA || !detailsB) return 0;
            let valA, valB;
            if (sortKey === 'id') {
                let rawA = detailsA.dataset.toteId; let rawB = detailsB.dataset.toteId;
                let numA_match = rawA.match(/\\d+/); let numB_match = rawB.match(/\\d+/);
                let numA = numA_match ? parseInt(numA_match[0], 10) : NaN;
                let numB = numB_match ? parseInt(numB_match[0], 10) : NaN;
                if (!isNaN(numA) && !isNaN(numB) && (rawA.replace(numA_match[0], "") === rawB.replace(numB_match[0], ""))) {
                    valA = numA; valB = numB;
                } else { valA = rawA.toLowerCase(); valB = rawB.toLowerCase(); }
            } else if (sortKey === 'items') {
                valA = parseInt(detailsA.dataset.items, 10); valB = parseInt(detailsB.dataset.items, 10);
            } else if (sortKey === 'utilization') {
                valA = parseFloat(detailsA.dataset.utilization); valB = parseFloat(detailsB.dataset.utilization);
            } else { return 0; }
            if (valA < valB) return sortOrder === 'asc' ? -1 : 1;
            if (valA > valB) return sortOrder === 'asc' ? 1 : -1;
            return 0;
        });
        toteWrappers.forEach(wrapper => container.appendChild(wrapper));
    }
    window.sortToteDetailsHTMLReport = sortToteDetailsHTMLReport;

    function filterTotesBySKU() {
        const filterInput = document.getElementById('skuFilterInput');
        const filterValue = filterInput.value.trim().toLowerCase();
        const toteWrappers = document.querySelectorAll('#tote-details-container .tote-wrapper');
        toteWrappers.forEach(wrapper => {
            const detailsElement = wrapper.querySelector('details');
            if (!detailsElement) return;
            if (filterValue === "") { wrapper.style.display = 'block'; return; }
            const skusData = detailsElement.dataset.skus;
            if (skusData) {
                const skusArray = skusData.toLowerCase().split(',');
                const matchFound = skusArray.some(sku => sku.includes(filterValue));
                wrapper.style.display = matchFound ? 'block' : 'none';
            } else { wrapper.style.display = 'none'; }
        });
    }
    window.filterTotesBySKU = filterTotesBySKU;
    </script>
    """

    # --- Methodology Section Content ---
    methodology_section_content_html = "<section id='methodology-overview'>"
    methodology_section_content_html += "<h2>Algorithm Overview and Methodology</h2>"
    methodology_html_formatted = METHODOLOGY_TEXT.replace("**", "<strong>") # Basic replacement for bold
    methodology_section_content_html += f"<div class='methodology-content'>{methodology_html_formatted}</div></section>"

    # --- Simulation Configuration Content ---
    sim_config_content_html = f"<p><span class='data-label'>Report Generated</span> <span class='data-value'>{generation_time}</span></p>"
    sim_config_content_html += f"<p><span class='data-label'>Tote Length</span> <span class='data-value'>{tote_config.get('TOTE_MAX_LENGTH', 'N/A')} mm</span></p>"
    sim_config_content_html += f"<p><span class='data-label'>Tote Width</span> <span class='data-value'>{tote_config.get('TOTE_MAX_WIDTH', 'N/A')} mm</span></p>"
    sim_config_content_html += f"<p><span class='data-label'>Tote Height</span> <span class='data-value'>{tote_config.get('TOTE_MAX_HEIGHT', 'N/A')} mm</span></p>"
    sim_config_content_html += f"<p><span class='data-label'>Height Map Resolution</span> <span class='data-value'>{tote_config.get('HEIGHT_MAP_RESOLUTION', 'N/A')} mm</span></p>"
    # Add other relevant config details using the same <p><span class='data-label'>...</span>...</p> format

    # --- Overall Packing Statistics Content ---
    overall_stats_content_html = f"<p><span class='data-label'>Total Totes Used</span> <span class='data-value'>{summary_stats_dict.get('total_totes_used', 'N/A')}</span></p>"
    avg_util_str = f"{summary_stats_dict.get('average_utilization', 0.0):.2f}%" if summary_stats_dict.get('average_utilization') is not None else "N/A"
    overall_stats_content_html += f"<p><span class='data-label'>Average Tote Utilization</span> <span class='data-value'>{avg_util_str}</span></p>"
    min_util_str = f"{summary_stats_dict.get('min_utilization', 0.0):.2f}%" if summary_stats_dict.get('min_utilization') is not None else "N/A"
    overall_stats_content_html += f"<p><span class='data-label'>Minimum Tote Utilization</span> <span class='data-value'>{min_util_str}</span></p>"
    median_util_str = f"{summary_stats_dict.get('median_utilization', 0.0):.2f}%" if summary_stats_dict.get('median_utilization') is not None else "N/A"
    overall_stats_content_html += f"<p><span class='data-label'>Median Tote Utilization</span> <span class='data-value'>{median_util_str}</span></p>"
    max_util_str = f"{summary_stats_dict.get('max_utilization', 0.0):.2f}%" if summary_stats_dict.get('max_utilization') is not None else "N/A"
    overall_stats_content_html += f"<p><span class='data-label'>Maximum Tote Utilization</span> <span class='data-value'>{max_util_str}</span></p>"
    overall_stats_content_html += f"<p><span class='data-label'>Total Cases Placed</span> <span class='data-value'>{summary_stats_dict.get('total_items_placed', 'N/A')}</span></p>"
    unplaced_count = summary_stats_dict.get('unplaced_items_count', 0)
    unplaced_badge_class = "badge-success" if unplaced_count == 0 else "badge-warning"
    unplaced_text = "None" if unplaced_count == 0 else f"{unplaced_count} <span class='badge {unplaced_badge_class}'>Review Needed</span>"
    overall_stats_content_html += f"<p><span class='data-label'>Cases That Did Not Fit</span> <span class='data-value'>{unplaced_text}</span></p>"
    avg_items_str = f"{summary_stats_dict.get('avg_items_per_tote', 0.0):.2f}"
    overall_stats_content_html += f"<p><span class='data-label'>Average Items per Tote</span> <span class='data-value'>{avg_items_str}</span></p>"
    # Add other overall stats using the same format

    # Histogram (optional, kept from previous logic)
    utilization_percentages_for_hist = [t.get('utilization_percent', 0.0) for t in all_totes_data if t.get('utilization_percent') is not None]
    if utilization_percentages_for_hist:
        overall_stats_content_html += "<h4 style='margin-top: 25px; margin-bottom: 10px;'>Distribution of Tote Utilizations:</h4>"
        try:
            fig_hist_report, ax_hist_report = plt.subplots(figsize=(6, 3.5))
            ax_hist_report.hist(utilization_percentages_for_hist, bins='auto', color='cornflowerblue', edgecolor='black', alpha=0.75)
            ax_hist_report.set_title('Tote Utilization Distribution', fontsize=11)
            ax_hist_report.set_xlabel('Utilization %', fontsize=9); ax_hist_report.set_ylabel('Number of Totes', fontsize=9)
            ax_hist_report.tick_params(axis='both', which='major', labelsize=8); ax_hist_report.grid(axis='y', linestyle='--', alpha=0.7)
            fig_hist_report.tight_layout(pad=0.8)
            buf_hist = io.BytesIO(); fig_hist_report.savefig(buf_hist, format='png', dpi=90); buf_hist.seek(0)
            img_base64_hist = base64.b64encode(buf_hist.getvalue()).decode('utf-8')
            overall_stats_content_html += f"<div class='tote-image-container' style='max-width: 550px; margin-left: auto; margin-right: auto;'><img src='data:image/png;base64,{img_base64_hist}' alt='Tote Utilization Histogram'></div>"
            plt.close(fig_hist_report)
        except Exception as e:
            overall_stats_content_html += f"<p><em>Error generating utilization histogram: {str(e)}</em></p>"
            if 'fig_hist_report' in locals() and fig_hist_report: plt.close(fig_hist_report)

    # --- Individual Tote Details Container Content ---
    tote_details_container_html = ""
    if all_totes_data:
        for tote_summary_info in all_totes_data:
            tote_id = tote_summary_info.get('id', 'Unknown Tote')
            items_in_tote_list = tote_summary_info.get('items', [])
            utilization = tote_summary_info.get('utilization_percent', 0.0)
            items_packed_count = len(items_in_tote_list)
            skus_in_tote_list = sorted(list(set(item.get('sku', '') for item in items_in_tote_list if item.get('sku'))))
            skus_in_tote_attr_str = ",".join(skus_in_tote_list)

            util_badge_class = "badge-success"
            if utilization < 70: util_badge_class = "badge-warning"
            if utilization < 50: util_badge_class = "badge-danger"
            
            # Data attributes for sorting (match JS: tote_id, item_count, volume_utilization)
            tote_details_container_html += f"<div class='tote-wrapper' data-tote-id='{tote_id}' data-item-count='{items_packed_count}' data-volume-utilization='{utilization:.2f}'>"
            tote_details_container_html += "<details class='tote-details-collapsible'>"
            
            # Summary with grid
            tote_details_container_html += f"<summary>Tote ID: {tote_id}"
            tote_details_container_html += "<div class='tote-summary-grid'>"
            tote_details_container_html += f"<span>Items: <strong>{items_packed_count}</strong></span>"
            tote_details_container_html += f"<span>Util: <strong class='badge {util_badge_class}'>{utilization:.2f}%</strong></span>"
            # Add total volume/weight to summary grid if available
            tote_details_container_html += "</div></summary>"
            
            # Details Content
            tote_details_container_html += "<div class='details-content'>"
            
            # 3D Visualization (if applicable)
            fig = visualization.generate_tote_figure(tote_summary_info)
            if fig:
                try:
                    buf = io.BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight', dpi=75); buf.seek(0)
                    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                    tote_details_container_html += f"<div class='tote-image-container'><img src='data:image/png;base64,{img_base64}' alt='Tote {tote_id} visualization'></div>"
                    plt.close(fig)
                except Exception as e:
                    tote_details_container_html += f"<p><em>Error generating image for Tote {tote_id}: {str(e)}</em></p>"
                    if fig: plt.close(fig)
            
            tote_details_container_html += f"<h4>Items in {tote_id}:</h4>"
            tote_details_container_html += f"<p><span class='data-label'>Total Items</span><span class='data-value'>{items_packed_count}</span></p>"
            # Add total weight, packed volume using data-label/data-value if available
            tote_details_container_html += f"<p><span class='data-label'>Volume Utilization</span><span class='data-value'><span class='badge {util_badge_class}'>{utilization:.2f}%</span></span></p>"

            # Items table
            tote_items_df = report_df[report_df['Tote ID'] == tote_id]
            if not tote_items_df.empty:
                # Prepare DataFrame for the new table structure
                display_item_df = tote_items_df[['SKU', 'Placed Length (mm)', 'Placed Width (mm)', 'Placed Height (mm)']].copy()
                display_item_df.rename(columns={
                    'Placed Length (mm)': 'Length', 'Placed Width (mm)': 'Width', 'Placed Height (mm)': 'Height'
                }, inplace=True)
                # Format dimensions - ensure numeric conversion before formatting
                for col in ['Length', 'Width', 'Height']:
                    display_item_df[col] = pd.to_numeric(display_item_df[col], errors='coerce')
                display_item_df['Dimensions (LxWxH)'] = display_item_df.apply(
                    lambda r: f"{r['Length']:.1f}x{r['Width']:.1f}x{r['Height']:.1f}" if pd.notna(r['Length']) and pd.notna(r['Width']) and pd.notna(r['Height']) else "N/A", 
                    axis=1
                )
                # Add Qty, Volume, Weight columns if available in report_df or calculable
                # Example: Aggregate by SKU if report_df has one row per item instance
                item_summary_table = display_item_df.groupby('SKU').agg(
                    Qty=('SKU', 'size'),
                    Dimensions=('Dimensions (LxWxH)', 'first') # Assumes dimensions are same for same SKU
                    # Add aggregation for Volume, Weight if data exists
                ).reset_index()
                
                # Rename columns for final display
                item_summary_table.rename(columns={'Dimensions': 'Dimensions (LxWxH)'}, inplace=True)
                
                # Select columns in desired order for the table
                final_cols = ['SKU', 'Qty', 'Dimensions (LxWxH)'] # Add Volume, Weight here
                item_summary_table = item_summary_table[[col for col in final_cols if col in item_summary_table.columns]]

                tote_details_container_html += item_summary_table.to_html(index=False, escape=False, classes="details-table", border=0)
            else:
                tote_details_container_html += "<p>No item details found for this tote.</p>"
            
            # Hidden paragraph for SKU filtering data
            tote_details_container_html += f"<p data-skus='{skus_in_tote_attr_str}' style='display:none;'>Internal SKU data</p>"
            tote_details_container_html += "</div></details></div>" # Close details-content, details, tote-wrapper
    else:
        tote_details_container_html += "<p>No tote data available for detailed view.</p>"

    # --- Unplaced Items Content ---
    unplaced_items_content_html = ""
    if unplaceable_items_log:
        unplaced_count_summary = len(unplaceable_items_log)
        # Calculate total volume/weight of unplaced if available in log
        
        unplaced_items_content_html += f"<details class='tote-details-collapsible unplaced-items-summary' {'open' if unplaceable_items_log else ''}>"
        unplaced_items_content_html += "<summary>Unplaced Items Summary"
        unplaced_items_content_html += "<div class='tote-summary-grid'>"
        unplaced_items_content_html += f"<span>Count: <strong>{unplaced_count_summary}</strong></span>"
        # Add total volume/weight summary if calculated
        unplaced_items_content_html += "</div></summary>"
        unplaced_items_content_html += "<div class='details-content'>"
        unplaced_items_content_html += "<p>The following items could not be placed:</p>"
        
        unplaced_df = pd.DataFrame(unplaceable_items_log)
        # Prepare DataFrame for the new table structure (SKU, Reason, Dimensions, Volume, Weight)
        display_cols_unplaced = {}
        if 'sku' in unplaced_df.columns: display_cols_unplaced['sku'] = 'SKU'
        if 'reason' in unplaced_df.columns: display_cols_unplaced['reason'] = 'Reason for Non-Placement'
        if 'dimensions' in unplaced_df.columns:
            def format_unplaced_dims(dims):
                 if isinstance(dims, (list, tuple)) and len(dims) == 3:
                     # Ensure numeric conversion before formatting
                     try: L, W, H = float(dims[0]), float(dims[1]), float(dims[2])
                     except (ValueError, TypeError): return "N/A"
                     return f"{L:.1f}x{W:.1f}x{H:.1f}"
                 return "N/A"
            unplaced_df['Formatted Dimensions'] = unplaced_df['dimensions'].apply(format_unplaced_dims)
            display_cols_unplaced['Formatted Dimensions'] = 'Dimensions (LxWxH)'
        # Add Volume, Weight if available in unplaceable_items_log

        cols_to_select = [col for col in display_cols_unplaced.keys() if col in unplaced_df.columns or col == 'Formatted Dimensions']
        if cols_to_select:
            unplaced_display_df = unplaced_df[cols_to_select].rename(columns=display_cols_unplaced)
            desired_order = ['SKU', 'Reason for Non-Placement', 'Dimensions (LxWxH)'] # Add Volume, Weight
            ordered_cols = [col for col in desired_order if col in unplaced_display_df.columns]
            remaining_cols = [col for col in unplaced_display_df.columns if col not in ordered_cols]
            unplaced_display_df = unplaced_display_df[ordered_cols + remaining_cols]
            unplaced_items_content_html += unplaced_display_df.to_html(index=False, escape=False, classes="details-table", border=0)
        else:
            unplaced_items_content_html += "<p>No detailed data for unplaced items.</p>"
        unplaced_items_content_html += "</div></details>"
    else:
        unplaced_items_content_html += "<p>All items were successfully placed or no items were attempted.</p>"


    # --- Load Template and Populate ---
    try:
        # Ensure the template path is correct relative to app.py
        with open("report_template.html", "r", encoding="utf-8") as f:
            template_content = f.read()
    except FileNotFoundError:
        st.error("Error: report_template.html not found in the current directory.")
        return "Error: report_template.html not found."
    except Exception as e:
        st.error(f"Error reading template: {str(e)}")
        return f"Error reading template: {str(e)}"

    # Replace placeholders in the template
    final_html = template_content.replace("{{ report_title }}", report_title)
    final_html = final_html.replace("{{ methodology_section_content }}", methodology_section_content_html)
    final_html = final_html.replace("{{ dynamic_simulation_configuration_content }}", sim_config_content_html)
    final_html = final_html.replace("{{ dynamic_overall_statistics_content }}", overall_stats_content_html)
    final_html = final_html.replace("{{ dynamic_individual_tote_details_container }}", tote_details_container_html)
    final_html = final_html.replace("{{ dynamic_unplaced_items_content }}", unplaced_items_content_html)
    final_html = final_html.replace("{{ footer_generation_time }}", generation_time)
    
    return final_html

# --- Sidebar Configuration ---
st.sidebar.header("Simulation Configuration")

# --- Tote Configuration ---
st.sidebar.subheader("Tote Dimensions (mm)")
tote_length_input = st.sidebar.number_input(
    "Tote Length", min_value=50, value=int(config.TOTE_MAX_LENGTH), step=10, key="tote_length" 
)
tote_width_input = st.sidebar.number_input(
    "Tote Width", min_value=50, value=int(config.TOTE_MAX_WIDTH), step=10, key="tote_width" 
)
tote_height_input = st.sidebar.number_input(
    "Tote Height", min_value=50, value=int(config.TOTE_MAX_HEIGHT), step=10, key="tote_height" 
)
height_map_resolution_input = st.sidebar.number_input(
    "Height Map Resolution (mm)", min_value=1, value=config.HEIGHT_MAP_RESOLUTION, step=1, key="height_map_resolution"
)

# --- Case Generation ---
st.sidebar.subheader("Case Data Source")
def reset_sim_ran_on_source_change():
    st.session_state.simulation_ran = False
    st.session_state.simulation_results = {'visualization_output_list': [], 'full_totes_summary_data': []}
    st.session_state.original_item_count = 0

case_data_source = st.sidebar.radio(
    "Select case data source:",
    ("Generate Random Cases", "Upload CSV File"),
    key="case_data_source",
    on_change=reset_sim_ran_on_source_change
)

if case_data_source == "Generate Random Cases":
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
else: 
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

    if uploaded_file is not None: 
        max_rows_input_val = st.sidebar.number_input(
            "Max rows to load from CSV (0 for all)",
            min_value=0,
            value=st.session_state.get('max_rows_csv', 0),
            step=100,
                key="max_rows_csv",
            help="Enter 0 or leave blank to load all rows. Headers are always read."
        )
        if max_rows_input_val > 0:
            st.session_state.csv_sampling_method = st.sidebar.radio(
                "Sampling method for subset:",
                ("Sequential", "Random"),
                index=0,
                key="csv_sampling_method_radio", # This key is fine, it's for the radio button itself
                help="Choose 'Sequential' to load the first X rows, or 'Random' to load X rows randomly from the entire file."
            )
            if st.session_state.csv_sampling_method == "Random": # Check the value from the radio button
                st.session_state.csv_random_seed = st.sidebar.number_input(
                    "CSV Random Seed",
                    value=st.session_state.get('csv_random_seed', 42), # Use existing or default
                    step=1,
                    key="csv_seed_input", # New key for this input
                    help="Seed for random sampling of CSV rows."
                )
        else:
            # If max_rows is 0, sampling method is not applicable, so ensure it's reset or default
            st.session_state.csv_sampling_method = 'Sequential' # Default back or could be None

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


# --- Run/Pause Simulation Buttons ---
run_col, pause_col = st.sidebar.columns(2)

if run_col.button("Run Simulation", key="run_button", type="primary", disabled=st.session_state.get('simulation_running', False)):
    st.session_state.simulation_ran = False
    st.session_state.simulation_running = True 
    st.session_state.simulation_paused = False
    st.session_state.simulation_progress = 0.0
    st.session_state.simulation_generator = None
    st.session_state.simulation_results = {'visualization_output_list': [], 'full_totes_summary_data': [], 'unplaceable_items_log': []} # Ensure log is reset
    st.session_state.intermediate_results = None
    st.session_state.original_item_count = 0 
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

    simulation_can_proceed = False
    current_input_cases = []
    st.session_state.current_tote_config_for_report = dynamic_tote_config 

    if case_data_source == "Generate Random Cases":
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
            st.session_state.simulation_running = False 

    elif case_data_source == "Upload CSV File":
        if uploaded_file is not None:
            len_col = st.session_state.column_mappings.get('length_col')
            wid_col = st.session_state.column_mappings.get('width_col')
            hei_col = st.session_state.column_mappings.get('height_col')
            sku_col_map = st.session_state.column_mappings.get('sku_col')

            if not all([len_col, wid_col, hei_col]):
                st.error("Column mapping incomplete. Please select columns for Length, Width, and Height.")
                st.session_state.simulation_running = False 
            else:
                try:
                    uploaded_file.seek(0)
                    columns_to_read = list(set(filter(None, [len_col, wid_col, hei_col, sku_col_map if sku_col_map != "Auto-generate SKU" else None])))
                    if not all(c in st.session_state.csv_headers for c in [len_col, wid_col, hei_col]):
                         st.error("Essential mapped columns not found in CSV headers.")
                         st.session_state.simulation_running = False 
                    else:
                        max_rows_val = st.session_state.get('max_rows_csv', 0)
                        nrows_to_load = int(max_rows_val) if max_rows_val > 0 else None
                        
                        sampling_method = st.session_state.get('csv_sampling_method', 'Sequential') if nrows_to_load else 'Sequential'

                        if nrows_to_load and sampling_method == "Random":
                            full_df_for_sampling = pd.read_csv(uploaded_file) 
                            if nrows_to_load >= len(full_df_for_sampling):
                                df = full_df_for_sampling[columns_to_read]
                                st.sidebar.info(f"Requested {nrows_to_load} random rows, but file only has {len(full_df_for_sampling)}. Loading all rows.")
                            else:
                                # Use the new csv_random_seed from session state
                                csv_seed_to_use = st.session_state.get('csv_random_seed', 42)
                                df = full_df_for_sampling.sample(n=nrows_to_load, random_state=csv_seed_to_use).reset_index(drop=True)
                                df = df[columns_to_read]
                            st.sidebar.info(f"Loaded {len(df)} rows randomly from CSV (seed: {csv_seed_to_use}).")
                        else:
                            df = pd.read_csv(uploaded_file, usecols=columns_to_read, nrows=nrows_to_load)
                            if nrows_to_load:
                                st.sidebar.info(f"Loaded first {len(df)} rows sequentially from CSV.")
                            else:
                                st.sidebar.info(f"Loaded all {len(df)} rows from CSV.")

                        if not all(pd.api.types.is_numeric_dtype(df[col]) for col in [len_col, wid_col, hei_col]):
                            st.error("Length, Width, Height columns must be numeric.")
                            st.session_state.simulation_running = False 
                        elif not all((df[col] > 0).all() for col in [len_col, wid_col, hei_col]):
                             st.error("Length, Width, Height values must be positive.")
                             st.session_state.simulation_running = False 
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
                    st.session_state.simulation_running = False 
        else:
            st.error("Please upload a CSV file.")
            st.session_state.simulation_running = False 

    if simulation_can_proceed and current_input_cases:
        st.session_state.simulation_generator = simulation.run_simulation_for_visualization_data(
            case_data_list=current_input_cases,
            current_tote_config=dynamic_tote_config
        )
        st.rerun() 
    elif not simulation_can_proceed:
        st.session_state.simulation_running = False 
        st.rerun()

pause_label = "Pause" if not st.session_state.get('simulation_paused') else "Resume"
if pause_col.button(pause_label, key="pause_button", disabled=not st.session_state.get('simulation_running', False)):
    st.session_state.simulation_paused = not st.session_state.simulation_paused
    st.session_state.status_message = "Simulation Paused." if st.session_state.simulation_paused else "Resuming simulation..."
    st.rerun()


# --- Simulation Progress Display Area ---
st.header("Simulation Process & Results")
progress_area = st.empty() 

# --- Consume Generator (if running and not paused) ---
if st.session_state.get('simulation_running') and not st.session_state.get('simulation_paused'):
    if st.session_state.simulation_generator:
        try:
            yielded_data = next(st.session_state.simulation_generator, None)

            if yielded_data:
                st.session_state.simulation_progress = yielded_data.get("progress", 0.0)
                st.session_state.status_message = yielded_data.get("status_message", "Processing...")
                st.session_state.intermediate_results = yielded_data 

                if yielded_data.get("is_final", False):
                    st.session_state.simulation_running = False
                    st.session_state.simulation_paused = False
                    st.session_state.simulation_ran = True
                    st.session_state.simulation_progress = 1.0 # Ensure 100%
                    st.session_state.simulation_results['visualization_output_list'] = yielded_data.get('intermediate_vis_data', [])
                    st.session_state.simulation_results['full_totes_summary_data'] = yielded_data.get('intermediate_totes_data', [])
                    st.session_state.simulation_results['unplaceable_items_log'] = yielded_data.get('unplaceable_log', [])
                    st.session_state.simulation_generator = None
                    st.success("Simulation finished!")
                    st.session_state.status_message = "Simulation Complete." # Override for clarity
                st.rerun()
            else: # Generator exhausted
                st.session_state.simulation_running = False
                st.session_state.simulation_paused = False
                st.session_state.simulation_ran = True
                st.session_state.simulation_generator = None
                st.session_state.simulation_progress = 1.0 # Explicitly set to 100%

                if st.session_state.intermediate_results: # These are the last results before exhaustion
                     st.session_state.simulation_results['visualization_output_list'] = st.session_state.intermediate_results.get('intermediate_vis_data', [])
                     st.session_state.simulation_results['full_totes_summary_data'] = st.session_state.intermediate_results.get('intermediate_totes_data', [])
                     st.session_state.simulation_results['unplaceable_items_log'] = st.session_state.intermediate_results.get('unplaceable_log', [])
                     # Set a clear completion message
                     st.session_state.status_message = "Simulation Complete (all items processed)."
                else:
                     # This case implies the generator finished without yielding any intermediate_results
                     # or that intermediate_results was None before this block.
                     st.session_state.status_message = "Simulation Complete (no data yielded)."
                     st.session_state.simulation_results = {'visualization_output_list': [], 'full_totes_summary_data': [], 'unplaceable_items_log': []}
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
    results_to_display = st.session_state.intermediate_results
    st.warning("Simulation Paused. Showing intermediate results.")
elif st.session_state.get('simulation_ran'):
    results_to_display = {
        'intermediate_totes_data': st.session_state.simulation_results['full_totes_summary_data'],
        'intermediate_vis_data': st.session_state.simulation_results['visualization_output_list'],
        'unplaceable_log': st.session_state.simulation_results.get('unplaceable_items_log', []) 
    }

if results_to_display:
    full_totes_summary_data = results_to_display.get('intermediate_totes_data', [])
    visualization_output_list = results_to_display.get('intermediate_vis_data', [])
    unplaceable_log_results = results_to_display.get('unplaceable_log', [])


    if full_totes_summary_data:
        st.subheader("Overall Simulation Statistics")

        utilization_percentages = [
            tote.get('utilization_percent', 0.0)
            for tote in full_totes_summary_data
            if tote.get('utilization_percent') is not None
        ]
        total_totes_used = len(full_totes_summary_data)
        total_items_placed_in_stats = sum(len(tote.get('items',[])) for tote in full_totes_summary_data)
        
        # Calculate unplaced_items_count based on the log from results_to_display
        unplaced_items_count = len(unplaceable_log_results)


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

        st.markdown("###### Packing Efficiency & Tote Performance")
        
        perf_data = [
            {"Metric": "Total Totes Used", "Value": str(total_totes_used)},
            {"Metric": "Total Cases Placed", "Value": str(total_items_placed_in_stats)},
            {"Metric": "Cases That Did Not Fit", "Value": str(unplaced_items_count)}, # Updated to use log length
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
        st.table(perf_df.set_index('Metric')) 

        if utilization_percentages:
            st.markdown("###### Distribution of Tote Utilizations") 
            fig_hist, ax_hist = plt.subplots(figsize=(4.0, 2.25)) 
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

        st.markdown("---") 
        st.markdown("###### SKU Distribution & Insights")
        
        sku_data = [
            {"Metric": "Total Unique SKUs Placed", "Value": str(num_unique_skus_placed)},
            {"Metric": "Avg. Totes per SKU", "Value": f"{avg_totes_per_sku:.2f}"},
            {"Metric": "Most Frequent SKU", "Value": most_frequent_sku_str},
        ]
        sku_df = pd.DataFrame(sku_data)
        st.table(sku_df.set_index('Metric'))

        st.markdown("---")
        st.markdown("###### Export Full Pack Report")
        if visualization_output_list: # This is the detailed item placement list
            report_data_items = [] # Renamed to avoid confusion with report_df for HTML
            for item_vis_data in visualization_output_list: 
                report_data_items.append({
                    "Tote ID": item_vis_data.get('tote_id', 'N/A'),
                    "SKU": item_vis_data.get('case_sku', 'N/A'),
                    "Placed Length (mm)": item_vis_data.get('placed_case_dims_mm', {}).get('length', 'N/A'),
                    "Placed Width (mm)": item_vis_data.get('placed_case_dims_mm', {}).get('width', 'N/A'),
                    "Placed Height (mm)": item_vis_data.get('placed_case_dims_mm', {}).get('height', 'N/A'),
                    "Position X (mm)": item_vis_data.get('position_mm', {}).get('x', 'N/A'),
                    "Position Y (mm)": item_vis_data.get('position_mm', {}).get('y', 'N/A'),
                    "Position Z (mm)": item_vis_data.get('position_mm', {}).get('z', 'N/A'),
                })
            report_df_for_export = pd.DataFrame(report_data_items) # This is the df for CSV and HTML items

            current_tote_config = st.session_state.get('current_tote_config_for_report', {})
            report_summary_stats = {
                "total_totes_used": total_totes_used,
                "total_items_placed": total_items_placed_in_stats,
                "unplaced_items_count": unplaced_items_count, # Use the accurate count
                "average_utilization": statistics.mean(utilization_percentages) if utilization_percentages else None,
                "min_utilization": min(utilization_percentages) if utilization_percentages else None,
                "median_utilization": statistics.median(utilization_percentages) if utilization_percentages else None,
                "max_utilization": max(utilization_percentages) if utilization_percentages else None,
                "avg_items_per_tote": avg_items_per_tote,
                "totes_with_one_item": totes_with_one_item,
                "percentage_single_case_totes": percentage_single_case_totes,
                "num_unique_skus_placed": num_unique_skus_placed,
                "avg_totes_per_sku": avg_totes_per_sku,
                "most_frequent_sku": most_frequent_sku_str,
            }
            
            csv_export = report_df_for_export.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Full Pack Report (CSV)",
                data=csv_export,
                file_name="bin_packing_report.csv",
                mime="text/csv",
                key="download_report_csv"
            )

            html_content_styled = generate_styled_html_report(
                report_df_for_export, # Pass the detailed item placement data
                report_summary_stats,
                current_tote_config,
                full_totes_summary_data, # This is the per-tote summary
                unplaceable_log_results 
            )
            html_export_styled = html_content_styled.encode('utf-8')
            st.download_button(
                label="Download Full Pack Report (HTML)",
                data=html_export_styled,
                file_name="bin_packing_report.html",
                mime="text/html",
                key="download_report_html_styled"
            )
        else:
            st.caption("No packing data available to generate a report.")

        st.divider() 

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
                        button_key = f"load_plot_button_{tote_id_str}_{i}" 
                        plot_loaded_session_key = f"plot_loaded_{tote_id_str}_{i}"

                        if st.button("Load/Refresh 3D View", key=button_key):
                            with st.spinner(f"Generating visualization for Tote {tote_id_str}..."):
                                fig = visualization.generate_tote_figure(tote_summary_info)
                                if fig:
                                    st.session_state[plot_loaded_session_key] = fig 
                                    st.pyplot(fig, clear_figure=True)
                                    plt.close(fig) 
                                else:
                                    st.warning("Could not generate visualization figure for this tote.")
                                    if plot_loaded_session_key in st.session_state:
                                        del st.session_state[plot_loaded_session_key] 
                        elif plot_loaded_session_key in st.session_state and st.session_state[plot_loaded_session_key]:
                             # If already loaded and button not pressed again, show cached plot
                             st.pyplot(st.session_state[plot_loaded_session_key], clear_figure=True)
                             # No need to close here as it's already closed after initial generation or should be managed carefully
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
                            pass # Keep as N/A or original string if conversion fails

                        st.dataframe(
                            item_details_df,
                            height=min(350, (len(item_details_list) + 1) * 35 + 3), # Dynamic height
                            use_container_width=True
                        )
                    else:
                        st.caption("No items in this tote.")
