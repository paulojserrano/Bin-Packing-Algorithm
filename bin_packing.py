import simulation
import visualization

# --- Main Execution Block ---
if __name__ == "__main__":
    num_random_cases = 10
    # Use generate_test_cases from the simulation module
    test_case_data = simulation.generate_test_cases(num_random_cases, seed=42) 
    
    # Use run_simulation_for_visualization_data from the simulation module
    simulation_visualization_data, full_totes_summary_data = simulation.run_simulation_for_visualization_data(test_case_data)
    
    print("\n--- Tote Utilization Summary (from simulation run) ---")
    if not full_totes_summary_data:
        print("No totes were processed in the simulation.")
    for tote_summary in full_totes_summary_data:
        print(f"Tote ID: {tote_summary['id']}, Items: {len(tote_summary['items'])}, Final Utilization: {tote_summary['utilization_percent']:.2f}%")

    if simulation_visualization_data:
        print("\nLaunching 3D Visualization...")
        # Use launch_visualization from the visualization module
        visualization.launch_visualization(simulation_visualization_data)
    else:
        print("\nNo items were placed in the simulation, so visualization will not be launched.")
