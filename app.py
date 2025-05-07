# app.py
import streamlit as st
import pandas as pd
import math # For calculations

# Import existing modules from the project
import simulation
import core_utils
import visualization # Now includes generate_tote_figure
import config # To get default values initially

# --- UI Configuration ---
st.set_page_config(layout="wide", page_title="Bin Packing Simulation")
st.title("Interactive Bin Packing Simulation")

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
# No longer need 'selected_tote_id_for_vis' as plots appear on expander open


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


case_data_source = st.sidebar.radio(
    "Select case data source:",
    ("Generate Random Cases", "Upload CSV File"),
    key="case_data_source",
    on_change=reset_sim_ran_on_source_change 
)

# test_case_data_input = [] # This variable is defined locally within the button click logic

if case_data_source == "Generate Random Cases":
    # If switching from CSV to Random, clear CSV-related session state
    if st.session_state.uploaded_file_name is not None: 
        st.session_state.csv_headers = []
        st.session_state.column_mappings = {k: None for k in st.session_state.column_mappings}
        st.session_state.uploaded_file_name = None
        # It's good practice to also reset the file uploader widget if possible,
        # though Streamlit handles this internally to some extent.
        # st.session_state.case_csv_uploader = None # This might be needed if uploader value persists undesirably

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
        "Upload Case Data CSV", type=["csv"], key="case_csv_uploader" # Ensure key is consistent
    )

    if uploaded_file is not None:
        # Process file only if it's a new file or different from the last one
        if st.session_state.uploaded_file_name != uploaded_file.name:
            st.session_state.uploaded_file_name = uploaded_file.name
            try:
                # Read only headers to get column names quickly
                df_headers = pd.read_csv(uploaded_file, nrows=0)
                st.session_state.csv_headers = df_headers.columns.tolist()
                # Reset previous mappings as the file has changed
                st.session_state.column_mappings = {k: None for k in st.session_state.column_mappings}
                uploaded_file.seek(0) # IMPORTANT: Reset file pointer for later full read
                st.sidebar.success(f"File '{uploaded_file.name}' loaded. Map columns below.")
            except Exception as e:
                st.sidebar.error(f"Error reading CSV headers: {e}")
                st.session_state.csv_headers = [] # Clear headers on error
                st.session_state.uploaded_file_name = None # Reset file name on error
    else:
        # If no file is uploaded (e.g., user removed it), reset CSV state
        if st.session_state.uploaded_file_name is not None:
            st.session_state.csv_headers = []
            st.session_state.column_mappings = {k: None for k in st.session_state.column_mappings}
            st.session_state.uploaded_file_name = None

    # Display column mapping UI only if headers are successfully read
    if st.session_state.csv_headers:
        st.sidebar.subheader("Map CSV Columns")
        def get_col_index(col_key, options_list, default_idx=0):
            """Helper to find index for selectbox, attempting to preserve selection."""
            prev_selection = st.session_state.column_mappings.get(col_key)
            if prev_selection and prev_selection in options_list:
                return options_list.index(prev_selection)
            # Ensure default_idx is within bounds of the options_list
            try: 
                return min(default_idx, len(options_list) -1) if options_list else 0
            except ValueError: # Should not happen if options_list is a list
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
            index=get_col_index('sku_col', sku_options, 0), key="map_sku" # Default to auto-generate
        )
    elif uploaded_file and not st.session_state.csv_headers: # File uploaded but headers not read
        st.sidebar.warning("Could not read columns. Check CSV format or re-upload.")


# --- Run Simulation Button ---
if st.sidebar.button("Run Simulation", key="run_button", type="primary"):
    st.session_state.simulation_ran = False # Reset before new run
    st.session_state.simulation_results = {'visualization_output_list': [], 'full_totes_summary_data': []}

    # Prepare dynamic tote configuration from sidebar inputs
    dynamic_tote_config = {
        "TOTE_MAX_LENGTH": int(tote_length_input),
        "TOTE_MAX_WIDTH": int(tote_width_input),
        "TOTE_MAX_HEIGHT": int(tote_height_input),
        "TOTE_MAX_VOLUME": int(tote_length_input * tote_width_input * tote_height_input),
        "HEIGHT_MAP_RESOLUTION": int(height_map_resolution_input),
        # Ensure grid dimensions are integers and at least 1
        "GRID_DIM_X": max(1, math.ceil(int(tote_length_input) / int(height_map_resolution_input))),
        "GRID_DIM_Y": max(1, math.ceil(int(tote_width_input) / int(height_map_resolution_input)))
    }

    simulation_can_proceed = False
    current_input_cases = [] # This will hold the list of case dictionaries

    # Prepare case data based on selected source
    if case_data_source == "Generate Random Cases":
        with st.spinner("Generating test cases..."):
            current_input_cases = simulation.generate_test_cases(
                num_cases=int(num_random_cases_input),
                seed=int(random_seed_input),
                current_tote_config=dynamic_tote_config # Pass tote config for better case generation
            )
        if current_input_cases:
            st.write(f"Generated {len(current_input_cases)} random cases.")
            simulation_can_proceed = True
        else:
            st.warning("No random cases were generated.")
    
    elif case_data_source == "Upload CSV File":
        if uploaded_file is not None: # Check if a file is actually uploaded
            # Retrieve mapped column names from session state
            len_col = st.session_state.column_mappings.get('length_col')
            wid_col = st.session_state.column_mappings.get('width_col')
            hei_col = st.session_state.column_mappings.get('height_col')
            sku_col_map = st.session_state.column_mappings.get('sku_col') # This can be "Auto-generate SKU" or a column name

            if not all([len_col, wid_col, hei_col]): # Check if essential columns are mapped
                st.error("Column mapping incomplete. Please select columns for Length, Width, and Height in the sidebar.")
            else:
                try:
                    uploaded_file.seek(0) # Ensure reading from the start of the file
                    df = pd.read_csv(uploaded_file)
                    
                    # Validate that mapped columns exist in the DataFrame and contain valid data
                    required_mapped_cols = {'Length': len_col, 'Width': wid_col, 'Height': hei_col}
                    valid_data = True
                    for std_name, actual_col in required_mapped_cols.items():
                        if actual_col not in df.columns:
                            st.error(f"Mapped column '{actual_col}' for {std_name} not found in CSV. Please check mappings or CSV file.")
                            valid_data = False; break
                        if not pd.api.types.is_numeric_dtype(df[actual_col]):
                            st.error(f"Column '{actual_col}' (mapped to {std_name}) must contain numeric values.")
                            valid_data = False; break
                        if not (df[actual_col] > 0).all(): # Ensure all dimension values are positive
                            st.error(f"All values in column '{actual_col}' (mapped to {std_name}) must be positive.")
                            valid_data = False; break
                    
                    if valid_data:
                        for index, row in df.iterrows():
                            sku_val = f"CSV_SKU_{index+1}" # Default SKU
                            # If a SKU column is mapped and it's not "Auto-generate SKU"
                            if sku_col_map and sku_col_map != "Auto-generate SKU":
                                if sku_col_map in df.columns: 
                                    sku_val = str(row[sku_col_map]) # Use the value from the mapped SKU column
                                else: # This case should ideally be caught if SKU col is mandatory and not found
                                      # For optional, this warning is fine.
                                    st.warning(f"SKU column '{sku_col_map}' selected in mapping but not found in CSV. Using auto-generated SKU for row {index+1}.")
                            
                            current_input_cases.append({
                                "sku": sku_val,
                                "length": float(row[len_col]),
                                "width": float(row[wid_col]),
                                "height": float(row[hei_col])
                            })
                        st.write(f"Loaded {len(current_input_cases)} cases from CSV using mapped columns.")
                        simulation_can_proceed = True
                except Exception as e:
                    st.error(f"Error processing CSV file with mapped columns: {e}")
        else:
            st.error("Please upload a CSV file when 'Upload CSV File' source is selected.")

    # Proceed with simulation if data is ready
    if simulation_can_proceed and current_input_cases:
        st.header("Simulation Process & Results")
        with st.spinner("Running packing simulation... This may take a moment."):
            sim_vis_data, full_totes_summary = simulation.run_simulation_for_visualization_data(
                case_data_list=current_input_cases,
                current_tote_config=dynamic_tote_config
            )
            # Store results in session state
            st.session_state.simulation_results['visualization_output_list'] = sim_vis_data
            st.session_state.simulation_results['full_totes_summary_data'] = full_totes_summary
            st.session_state.simulation_ran = True # Mark simulation as successfully run
            st.success("Simulation finished!")
    elif simulation_can_proceed and not current_input_cases: # Proceed flag was true but no cases
         st.warning("No case data was generated or loaded. Please check your inputs.")
    elif not simulation_can_proceed and (case_data_source == "Upload CSV File" and not uploaded_file and st.session_state.column_mappings.get('length_col')):
        # This condition means CSV source is chosen, mappings might be there from a previous file, but no current file.
        # The error "Please upload a CSV file..." is already handled above if uploaded_file is None.
        pass 
    elif not simulation_can_proceed and current_input_cases: # Should not happen if proceed is false due to earlier errors
        st.error("Simulation cannot proceed despite having case data. Check error messages above in the main panel or sidebar.")


# --- Display Results (only if simulation has run successfully) ---
if st.session_state.simulation_ran:
    st.subheader("Tote Utilization Summary & Visualization")
    full_totes_summary_data = st.session_state.simulation_results['full_totes_summary_data']

    if not full_totes_summary_data:
        st.write("No totes were used or processed in the simulation (e.g., no items to pack or items too large for the configured tote).")
    else:
        # Iterate through each tote's summary data to display it
        for i, tote_summary_info in enumerate(full_totes_summary_data):
            tote_id_str = str(tote_summary_info.get('id', f'Tote_{i+1}')) # Use a fallback ID if 'id' is missing
            items_in_tote = tote_summary_info.get('items', []) # List of item dicts in this tote
            items_packed_count = len(items_in_tote)
            utilization = tote_summary_info.get('utilization_percent', 0.0)

            # Expander for each tote. It will be collapsed by default.
            # When the user clicks to expand it, the content (plot and SKU table) will be rendered.
            with st.expander(f"Tote ID: {tote_id_str} | Items: {items_packed_count} | Utilization: {utilization:.2f}%", expanded=False):
                
                # Use columns to place plot and SKU table side-by-side
                col1, col2 = st.columns([2, 1]) # Plot in col1 (wider), SKU table in col2

                with col1: # Visualization Column
                    st.markdown("###### 3D View") # Small header for the plot
                    if items_packed_count > 0: # Only attempt to plot if there are items
                        with st.spinner(f"Generating visualization for Tote {tote_id_str}..."):
                            # Pass the specific tote's data to the generation function
                            # This data comes from `full_totes_summary_data` which should have all item details
                            fig = visualization.generate_tote_figure(tote_summary_info) 
                            if fig:
                                st.pyplot(fig, clear_figure=True) # clear_figure is important for Matplotlib in Streamlit
                            else:
                                st.warning("Could not generate visualization figure for this tote.")
                    else:
                        st.caption("No items in this tote to visualize.")
                
                with col2: # SKU Table Column
                    st.markdown("###### Packed SKUs") # Small header for the table
                    if items_in_tote:
                        # Create a list of dictionaries for the DataFrame, extracting only SKUs
                        sku_list_for_df = [{"SKU": item.get('sku', 'N/A')} for item in items_in_tote]
                        sku_df = pd.DataFrame(sku_list_for_df)
                        # Display the DataFrame. Adjust height dynamically or set a max.
                        st.dataframe(sku_df, height=min(300, (len(sku_list_for_df) + 1) * 35 + 3), use_container_width=True) # +3 for header padding
                    else:
                        st.caption("No items in this tote.")
else:
    # This message shows if simulation_ran is False (e.g., on first load or after changing source)
    st.info("Configure parameters in the sidebar and click 'Run Simulation' to begin and see results.")

