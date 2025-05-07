# app.py
import streamlit as st
import pandas as pd
import math # For calculations

# Import existing modules from the project
import simulation
import core_utils
import visualization
import config # To get default values initially

# --- UI Configuration ---
st.set_page_config(layout="wide", page_title="Bin Packing Simulation")
st.title("Interactive Bin Packing Simulation")

st.sidebar.header("Simulation Configuration")

# --- Tote Configuration ---
st.sidebar.subheader("Tote Dimensions (mm)")
# Use default values from config.py initially
tote_length_input = st.sidebar.number_input(
    "Tote Length",
    min_value=50,
    value=config.TOTE_MAX_LENGTH, # Default from config.py
    step=10,
    key="tote_length"
)
tote_width_input = st.sidebar.number_input(
    "Tote Width",
    min_value=50,
    value=config.TOTE_MAX_WIDTH, # Default from config.py
    step=10,
    key="tote_width"
)
tote_height_input = st.sidebar.number_input(
    "Tote Height",
    min_value=50,
    value=config.TOTE_MAX_HEIGHT, # Default from config.py
    step=10,
    key="tote_height"
)
height_map_resolution_input = st.sidebar.number_input(
    "Height Map Resolution (mm)",
    min_value=1,
    value=config.HEIGHT_MAP_RESOLUTION, # Default from config.py
    step=1,
    key="height_map_resolution"
)

# --- Case Generation ---
st.sidebar.subheader("Case Data Source")
case_data_source = st.sidebar.radio(
    "Select case data source:",
    ("Generate Random Cases", "Upload CSV File"),
    key="case_data_source"
)

test_case_data = [] # Initialize test_case_data

if case_data_source == "Generate Random Cases":
    st.sidebar.subheader("Random Case Generation")
    num_random_cases_input = st.sidebar.number_input(
        "Number of Random Cases to Generate",
        min_value=1,
        value=10,
        step=1,
        key="num_cases"
    )
    random_seed_input = st.sidebar.number_input(
        "Random Seed for Case Generation",
        value=42,
        step=1,
        key="random_seed"
    )
else: # "Upload CSV File"
    st.sidebar.subheader("Upload Case Data")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Case Data CSV",
        type=["csv"],
        key="case_csv_uploader"
    )

# Button to run the simulation
if st.sidebar.button("Run Simulation", key="run_button"):
    # --- Prepare Dynamic Configuration for the Simulation ---
    dynamic_tote_config = {
        "TOTE_MAX_LENGTH": int(tote_length_input),
        "TOTE_MAX_WIDTH": int(tote_width_input),
        "TOTE_MAX_HEIGHT": int(tote_height_input),
        "TOTE_MAX_VOLUME": int(tote_length_input * tote_width_input * tote_height_input),
        "HEIGHT_MAP_RESOLUTION": int(height_map_resolution_input),
        "GRID_DIM_X": math.ceil(tote_length_input / height_map_resolution_input),
        "GRID_DIM_Y": math.ceil(tote_width_input / height_map_resolution_input)
    }

    st.header("Simulation Process & Results")
    simulation_can_proceed = False

    if case_data_source == "Generate Random Cases":
        with st.spinner("Generating test cases..."):
            test_case_data = simulation.generate_test_cases(
                num_cases=int(num_random_cases_input),
                seed=int(random_seed_input),
                current_tote_config=dynamic_tote_config
            )
            st.write(f"Generated {len(test_case_data)} random test cases.")
            if test_case_data:
                st.expander("View Generated Test Cases").json([vars(case) if hasattr(case, '__dict__') else case for case in test_case_data])
            simulation_can_proceed = True
    
    elif case_data_source == "Upload CSV File":
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                # Normalize column names to lowercase for easier checking
                df.columns = [col.lower() for col in df.columns]
                
                required_cols = ['length', 'width', 'height']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    st.error(f"Uploaded CSV is missing required columns: {', '.join(missing_cols)}. Please ensure 'length', 'width', and 'height' columns are present.")
                else:
                    # Validate data types and positive values
                    valid_data = True
                    for col in required_cols:
                        if not pd.api.types.is_numeric_dtype(df[col]):
                            st.error(f"Column '{col}' must contain numeric values.")
                            valid_data = False
                            break
                        if not (df[col] > 0).all():
                            st.error(f"All values in column '{col}' must be positive.")
                            valid_data = False
                            break
                    
                    if valid_data:
                        test_case_data = []
                        for index, row in df.iterrows():
                            sku = row.get("sku", f"CSV_SKU_{index+1}")
                            test_case_data.append({
                                "sku": sku,
                                "length": float(row["length"]),
                                "width": float(row["width"]),
                                "height": float(row["height"])
                            })
                        st.write(f"Loaded {len(test_case_data)} cases from CSV.")
                        st.expander("View Uploaded Case Data (First 5 Rows)").dataframe(df.head())
                        simulation_can_proceed = True

            except Exception as e:
                st.error(f"Error processing CSV file: {e}")
        else:
            st.error("Please upload a CSV file when 'Upload CSV File' source is selected.")

    if simulation_can_proceed and test_case_data:
        with st.spinner("Running packing simulation..."):
            simulation_visualization_data, full_totes_summary_data = \
                simulation.run_simulation_for_visualization_data(
                    case_data_list=test_case_data,
                    current_tote_config=dynamic_tote_config
                )
            st.write("Simulation finished.")

        # --- Display Summary ---
    st.subheader("Tote Utilization Summary")
    if not full_totes_summary_data:
        st.write("No totes were used or processed in the simulation.")
    else:
        summary_df_list = []
        for tote_summary in full_totes_summary_data:
            summary_df_list.append({
                "Tote ID": tote_summary.get('id', 'N/A'),
                "Items Packed": len(tote_summary.get('items', [])),
                "Final Utilization (%)": f"{tote_summary.get('utilization_percent', 0.0):.2f}"
            })
        summary_df = pd.DataFrame(summary_df_list)
        st.table(summary_df)

    # --- Display Visualization ---
    st.subheader("3D Visualization")
    if simulation_visualization_data:
        st.info("""
            The 3D visualization will attempt to launch in a separate interactive Matplotlib window.
            For direct embedding within Streamlit, `visualization.py` would require significant refactoring
            to return a Matplotlib Figure object and manage state via Streamlit widgets.
        """)
        try:
            # This will open a new window as per current visualization.py design
            visualization.launch_visualization(simulation_visualization_data)
            st.success("Visualization launched in a separate window (if successful and data available). Check your taskbar or other open windows.")
        except Exception as e:
            st.error(f"Could not launch visualization: {e}")
    else:
        st.write("No items were placed in the simulation, so no visualization data is available to display.")
else:
    st.info("Adjust parameters in the sidebar and click 'Run Simulation' to begin.")
