# visualization.py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.widgets import Button # Keep for old function if retained
import numpy as np

# --- Visualization Helper Functions (plot_cube, get_distinct_colors) ---
def plot_cube(ax, origin, dimensions, color='blue', alpha=0.1, label=None, edges=True, face_alpha_override=None):
    """
    Plots a 3D cube (representing a tote or a case) on the given Matplotlib 3D axis.

    Args:
        ax (matplotlib.axes.Axes3D): The 3D axes to plot on.
        origin (list or tuple): The [x, y, z] coordinates of the cube's origin.
        dimensions (list or tuple): The [length, width, height] of the cube.
        color (str, optional): The color of the cube. Defaults to 'blue'.
        alpha (float, optional): The transparency of the cube's faces. Defaults to 0.1.
        label (str, optional): A label for the cube, displayed as text. Defaults to None.
        edges (bool, optional): Whether to draw black edges for the cube. Defaults to True.
        face_alpha_override (float, optional): Specific alpha for faces, overriding the main alpha. Defaults to None.
    """
    x, y, z = origin
    dx, dy, dz = dimensions

    # Ensure dimensions are positive for plotting
    dx, dy, dz = max(1e-9, dx), max(1e-9, dy), max(1e-9, dz)


    vertices = [
        [x, y, z], [x + dx, y, z], [x + dx, y + dy, z], [x, y + dy, z],
        [x, y, z + dz], [x + dx, y, z + dz], [x + dx, y + dy, z + dz], [x, y + dy, z + dz]
    ]
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]], [vertices[4], vertices[5], vertices[6], vertices[7]],
        [vertices[0], vertices[1], vertices[5], vertices[4]], [vertices[3], vertices[2], vertices[6], vertices[7]],
        [vertices[1], vertices[5], vertices[6], vertices[2]], [vertices[0], vertices[4], vertices[7], vertices[3]]
    ]
    face_alpha_actual = alpha if face_alpha_override is None else face_alpha_override
    poly3d = Poly3DCollection(faces, facecolors=color, linewidths=1, edgecolors='k' if edges else color, alpha=face_alpha_actual)
    ax.add_collection3d(poly3d)
    if label:
        # Adjust text position for better visibility
        # Access tote_dims through a global variable if set by the calling function.
        # This is a workaround; ideally, relative text positioning parameters would be passed.
        tote_height_for_label_offset = getattr(plot_cube, 'current_tote_dims', [0,0,100])[2] # Default to 100 if not set
        ax.text(x + dx / 2, y + dy / 2, z + dz + (0.02 * tote_height_for_label_offset), label,
                color='black', ha='center', va='bottom', fontsize=6, # Reduced fontsize for smaller plot
                bbox=dict(facecolor='white', alpha=0.6, pad=0.1, boxstyle='round,pad=0.2'))

def get_distinct_colors(n):
    """
    Generates a list of n distinct colors for visualizing different items.

    Args:
        n (int): The number of distinct colors needed.

    Returns:
        list: A list of color names or RGBA tuples.
    """
    if n <= 0: return []
    predefined = ['deepskyblue', 'salmon', 'lightgreen', 'gold', 'orchid', 'lightcoral', 'mediumturquoise', 'orange', 'lime', 'pink',
                  'cyan', 'magenta', 'yellow', 'teal', 'lavender', 'sienna', 'palegreen', 'tan']
    if n <= len(predefined): return predefined[:n]
    
    try:
        cmap = plt.cm.get_cmap('viridis', n) 
        return [cmap(i / (n -1 if n > 1 else 1)) for i in range(n)]
    except Exception: 
        base_colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        return [base_colors[i % len(base_colors)] for i in range(n)]


# --- NEW FUNCTION for Streamlit Embedding ---
def generate_tote_figure(tote_data_for_vis):
    """
    Generates a Matplotlib Figure object for a single tote and its contents,
    suitable for embedding in Streamlit using st.pyplot().

    Args:
        tote_data_for_vis (dict): Data for one tote.
                                  Expected keys: 'id', 'max_length', 'max_width',
                                  'max_height', 'height_map_resolution', 'items' (list),
                                  'utilization_percent'.
                                  Each item in 'items' needs: 'sku', 'chosen_orientation_dims',
                                  'position_in_tote_grid', 'placement_z_level'.

    Returns:
        matplotlib.figure.Figure: The generated figure, or a figure with an error message.
    """
    if not tote_data_for_vis or not isinstance(tote_data_for_vis, dict):
        fig_err = plt.figure(figsize=(4, 3)) # Smaller error figure
        ax_err = fig_err.add_subplot(111)
        ax_err.text(0.5, 0.5, "Invalid tote data.", ha='center', va='center', fontsize=10)
        ax_err.axis('off')
        return fig_err

    tote_id = tote_data_for_vis.get('id', 'Unknown')
    
    current_tote_dims = [
        float(tote_data_for_vis.get('max_length', 100.0)),
        float(tote_data_for_vis.get('max_width', 100.0)),
        float(tote_data_for_vis.get('max_height', 100.0))
    ]
    # Set current tote dimensions as an attribute of plot_cube for label positioning
    # This is a workaround for not being able to pass it directly in this structure easily
    plot_cube.current_tote_dims = current_tote_dims


    items_in_tote = tote_data_for_vis.get('items', [])
    h_map_res = float(tote_data_for_vis.get('height_map_resolution', 10.0)) 
    utilization = float(tote_data_for_vis.get('utilization_percent', 0.0)) 

    if not all(d > 0 for d in current_tote_dims):
        fig_err = plt.figure(figsize=(4, 3))
        ax_err = fig_err.add_subplot(111)
        ax_err.text(0.5, 0.5, f"Tote {tote_id} invalid dims.", ha='center', va='center', fontsize=10)
        ax_err.axis('off')
        return fig_err

    # Reduced figsize for a smaller plot (approx. 50% linear reduction -> 25% area)
    fig = plt.figure(figsize=(4.5, 4.0)) 
    ax = fig.add_subplot(111, projection='3d')

    plot_cube(ax, [0, 0, 0], current_tote_dims, color='lightgrey', alpha=0.05, edges=True, face_alpha_override=0.03)

    item_colors = get_distinct_colors(len(items_in_tote))

    for i, item_vis in enumerate(items_in_tote):
        item_dims_raw = item_vis.get('chosen_orientation_dims')
        item_pos_grid = item_vis.get('position_in_tote_grid') 
        item_z_level_raw = item_vis.get('placement_z_level')
        item_sku = item_vis.get('sku', f'Item_{i+1}')

        if not item_dims_raw or not item_pos_grid or item_z_level_raw is None:
            print(f"Warning: Skipping item {item_sku} in tote {tote_id} due to missing data.")
            continue
        
        item_dims = [float(d) for d in item_dims_raw]
        item_z_level = float(item_z_level_raw)
        actual_x = float(item_pos_grid[0]) * h_map_res
        actual_y = float(item_pos_grid[1]) * h_map_res
        actual_z = item_z_level
        item_origin = [actual_x, actual_y, actual_z]
        
        color_idx = i % len(item_colors) if item_colors else 0
        current_item_color = item_colors[color_idx] if item_colors else 'purple'
        
        plot_cube(ax, item_origin, item_dims, color=current_item_color, alpha=0.7, label=item_sku) # Reduced alpha for less clutter

    ax.set_title(f"Tote {tote_id} (Util: {utilization:.2f}%)", fontsize=9) # Smaller font
    ax.set_xlabel("L (mm)", fontsize=7) # Shorter labels, smaller font
    ax.set_ylabel("W (mm)", fontsize=7)
    ax.set_zlabel("H (mm)", fontsize=7)
    ax.tick_params(axis='both', which='major', labelsize=6) # Smaller tick labels

    ax.set_xlim([0, current_tote_dims[0]])
    ax.set_ylim([0, current_tote_dims[1]])
    ax.set_zlim([0, current_tote_dims[2]])

    try:
        x_range = np.ptp(ax.get_xlim())
        y_range = np.ptp(ax.get_ylim())
        z_range = np.ptp(ax.get_zlim())
        if x_range > 0 and y_range > 0 and z_range > 0:
             ax.set_box_aspect([x_range, y_range, z_range])
        else: 
            ax.set_box_aspect([1,1,1]) 
            # print(f"Warning: Tote {tote_id} has zero range in one or more dimensions. Using default aspect.")
    except AttributeError:
        # print("Note: ax.set_box_aspect is not available. Plot may appear distorted.")
        pass # Suppress print for cleaner UI
    except Exception as e:
        # print(f"Error setting box aspect for Tote {tote_id}: {e}")
        pass

    ax.view_init(elev=20, azim=125) # Slightly adjusted view
    try:
        fig.tight_layout(pad=0.5) # Reduced padding
    except Exception:
        # print("Warning: fig.tight_layout() failed.")
        pass
    
    # Clean up the attribute after use if desired, though not strictly necessary here
    # if hasattr(plot_cube, 'current_tote_dims'):
    # delattr(plot_cube, 'current_tote_dims')

    return fig


# --- Existing Interactive Visualization Application State and Logic (can be kept or removed) ---
# These are for the standalone Matplotlib window and are not used by the new generate_tote_figure
vis_fig, vis_ax = None, None
vis_tote_ids_list = []
# ... (rest of the old code for launch_visualization can remain if dual functionality is desired)
# ... or be removed if only inline display is needed. For brevity, I'll assume it can remain for now.

def format_simulation_data_for_visualization(simulation_visualization_data):
    """Converts simulation output to the nested structure needed by the OLD interactive visualizer."""
    formatted_data = {}
    if not simulation_visualization_data:
        return formatted_data
    for item_sim_data in simulation_visualization_data:
        tote_id = item_sim_data['tote_id']
        if tote_id not in formatted_data:
            formatted_data[tote_id] = {
                'dims': [
                    item_sim_data['tote_dimensions_mm']['length'],
                    item_sim_data['tote_dimensions_mm']['width'],
                    item_sim_data['tote_dimensions_mm']['height']
                ],
                'items': []
            }
        formatted_data[tote_id]['items'].append({
            'sku': item_sim_data['case_sku'],
            'dims': [
                item_sim_data['placed_case_dims_mm']['length'],
                item_sim_data['placed_case_dims_mm']['width'],
                item_sim_data['placed_case_dims_mm']['height']
            ],
            'pos': [
                item_sim_data['position_mm']['x'],
                item_sim_data['position_mm']['y'],
                item_sim_data['position_mm']['z']
            ],
            'current_utilization': item_sim_data.get('current_tote_utilization_percent', 0.0)
        })
    return formatted_data

def update_visualization_display():
    global vis_ax, vis_current_tote_idx, vis_current_step_in_tote, vis_case_colors_for_current_tote, vis_button_prev, vis_button_next, vis_fig
    if not vis_parsed_totes_data or not vis_tote_ids_list: # ... (rest of function)
        if vis_ax: vis_ax.clear(); vis_ax.text(0.5,0.5,0.5,"No data.", ha='center'); vis_fig.canvas.draw_idle()
        return
    vis_ax.clear()
    current_tote_id_vis = vis_tote_ids_list[vis_current_tote_idx]
    tote_data_vis = vis_parsed_totes_data[current_tote_id_vis]
    plot_cube.current_tote_dims = tote_data_vis['dims'] # For label context
    # ... (rest of this function largely unchanged from previous version)
    tote_dims_vis = tote_data_vis['dims']
    items_in_tote_vis = tote_data_vis['items']
    if not vis_case_colors_for_current_tote and items_in_tote_vis:
        vis_case_colors_for_current_tote = get_distinct_colors(len(items_in_tote_vis))
    plot_cube(vis_ax, [0,0,0], tote_dims_vis, color='lightgrey', alpha=0.05)
    title_str = f"Tote {current_tote_id_vis} - "
    current_step_utilization = 0.0
    if vis_current_step_in_tote == -1: title_str += "Empty"
    else:
        for i in range(vis_current_step_in_tote + 1):
            if i < len(items_in_tote_vis):
                item = items_in_tote_vis[i]
                plot_cube(vis_ax, item['pos'], item['dims'], color=vis_case_colors_for_current_tote[i % len(vis_case_colors_for_current_tote)], alpha=0.8, label=item['sku'])
        if 0 <= vis_current_step_in_tote < len(items_in_tote_vis):
            current_step_utilization = items_in_tote_vis[vis_current_step_in_tote].get('current_utilization', 0.0)
            title_str += f"Added {items_in_tote_vis[vis_current_step_in_tote]['sku']}"
    title_str += f" (Util: {current_step_utilization:.2f}%)"
    vis_ax.set_title(title_str)
    # ... (set limits, labels, aspect, view_init, draw_idle, button states)
    vis_ax.set_xlim([0, tote_dims_vis[0]]); vis_ax.set_ylim([0, tote_dims_vis[1]]); vis_ax.set_zlim([0, tote_dims_vis[2]])
    try: vis_ax.set_box_aspect([np.ptp(vis_ax.get_xlim()), np.ptp(vis_ax.get_ylim()), np.ptp(vis_ax.get_zlim())])
    except: pass
    vis_ax.view_init(elev=25, azim=135)
    vis_fig.canvas.draw_idle()


def on_next_visualization(event): # Unchanged
    global vis_current_tote_idx, vis_current_step_in_tote, vis_case_colors_for_current_tote
    if not vis_tote_ids_list: return
    current_tote_id_vis = vis_tote_ids_list[vis_current_tote_idx]
    items_in_current_tote_vis = vis_parsed_totes_data[current_tote_id_vis]['items']
    max_step_for_current_tote_vis = len(items_in_current_tote_vis) - 1
    if vis_current_step_in_tote < max_step_for_current_tote_vis: vis_current_step_in_tote += 1
    else:
        if vis_current_tote_idx < len(vis_tote_ids_list) - 1:
            vis_current_tote_idx += 1; vis_current_step_in_tote = -1
            new_tote_id_vis = vis_tote_ids_list[vis_current_tote_idx]
            vis_case_colors_for_current_tote = get_distinct_colors(len(vis_parsed_totes_data[new_tote_id_vis]['items']))
    update_visualization_display()

def on_prev_visualization(event): # Unchanged
    global vis_current_tote_idx, vis_current_step_in_tote, vis_case_colors_for_current_tote
    if not vis_tote_ids_list: return
    if vis_current_step_in_tote > -1: vis_current_step_in_tote -= 1
    else:
        if vis_current_tote_idx > 0:
            vis_current_tote_idx -= 1
            new_tote_id_vis = vis_tote_ids_list[vis_current_tote_idx]
            items_in_new_tote_vis = vis_parsed_totes_data[new_tote_id_vis]['items']
            vis_current_step_in_tote = len(items_in_new_tote_vis) - 1
            vis_case_colors_for_current_tote = get_distinct_colors(len(items_in_new_tote_vis))
    update_visualization_display()

def launch_visualization(simulation_data_for_vis): # Largely unchanged
    global vis_fig, vis_ax, vis_tote_ids_list, vis_current_tote_idx, vis_current_step_in_tote, vis_button_prev, vis_button_next, vis_case_colors_for_current_tote, vis_parsed_totes_data
    vis_parsed_totes_data = format_simulation_data_for_visualization(simulation_data_for_vis)
    if not vis_parsed_totes_data: print("No data for old visualizer."); return
    vis_tote_ids_list = sorted(list(vis_parsed_totes_data.keys()))
    if not vis_tote_ids_list: print("No tote IDs for old visualizer."); return
    vis_current_tote_idx = 0; vis_current_step_in_tote = -1
    first_tote_id_vis = vis_tote_ids_list[vis_current_tote_idx]
    vis_case_colors_for_current_tote = get_distinct_colors(len(vis_parsed_totes_data[first_tote_id_vis]['items']))
    if plt.get_fignums(): vis_fig = plt.gcf(); vis_fig.clf()
    else: vis_fig = plt.figure(figsize=(10, 8.5))
    vis_ax = vis_fig.add_axes([0.05, 0.15, 0.9, 0.8], projection='3d')
    ax_prev_vis = vis_fig.add_axes([0.3, 0.05, 0.15, 0.075]); vis_button_prev = Button(ax_prev_vis, 'Previous'); vis_button_prev.on_clicked(on_prev_visualization)
    ax_next_vis = vis_fig.add_axes([0.55, 0.05, 0.15, 0.075]); vis_button_next = Button(ax_next_vis, 'Next'); vis_button_next.on_clicked(on_next_visualization)
    update_visualization_display()
    plt.show()

