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
st.sidebar.subheader("Case Generation")
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

# Button to run the simulation
if st.sidebar.button("Run Simulation", key="run_button"):
    # --- Prepare Dynamic Configuration for the Simulation ---
    # This dictionary will be passed to the simulation functions.
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
    with st.spinner("Generating test cases..."):
        # Pass the dynamic_tote_config to generate_test_cases
        test_case_data = simulation.generate_test_cases(
            num_cases=int(num_random_cases_input),
            seed=int(random_seed_input),
            current_tote_config=dynamic_tote_config # New argument
        )
        st.write(f"Generated {len(test_case_data)} test cases.")
        if test_case_data:
            st.expander("View Generated Test Cases").json([vars(case) if hasattr(case, '__dict__') else case for case in test_case_data])


    with st.spinner("Running packing simulation..."):
        # Pass the dynamic_tote_config to run_simulation_for_visualization_data
        simulation_visualization_data, full_totes_summary_data = \
            simulation.run_simulation_for_visualization_data(
                case_data_list=test_case_data,
                current_tote_config=dynamic_tote_config # New argument
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
