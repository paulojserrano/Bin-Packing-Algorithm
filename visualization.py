import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.widgets import Button
import numpy as np

# --- Visualization Helper Functions ---
def plot_cube(ax, origin, dimensions, color='blue', alpha=0.1, label=None, edges=True, face_alpha_override=None):
    x, y, z = origin
    dx, dy, dz = dimensions
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
        ax.text(x + dx / 2, y + dy / 2, z + dz + 10, label,
                color='black', ha='center', va='bottom', fontsize=7,
                bbox=dict(facecolor='white', alpha=0.7, pad=0.2, boxstyle='round,pad=0.3'))

def get_distinct_colors(n):
    if n <= 0: return []
    predefined = ['deepskyblue', 'salmon', 'lightgreen', 'gold', 'orchid', 'lightcoral', 'mediumturquoise', 'orange', 'lime', 'pink']
    if n <= len(predefined): return predefined[:n]
    cmap = plt.cm.get_cmap('viridis', n)
    return [cmap(i / (n -1 if n > 1 else 1)) for i in range(n)]

# --- Interactive Visualization Application State and Logic ---
vis_fig, vis_ax = None, None
vis_tote_ids_list = []
vis_current_tote_idx = 0
vis_current_step_in_tote = -1
vis_case_colors_for_current_tote = []
vis_button_prev = None
vis_button_next = None
vis_parsed_totes_data = {}

def format_simulation_data_for_visualization(simulation_visualization_data):
    """Converts simulation output to the nested structure needed by the visualizer."""
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
    """Updates the Matplotlib 3D plot for the visualization, including utilization in title."""
    global vis_ax, vis_current_tote_idx, vis_current_step_in_tote, vis_case_colors_for_current_tote, vis_button_prev, vis_button_next, vis_fig

    if not vis_parsed_totes_data or not vis_tote_ids_list:
        if vis_ax:
            vis_ax.clear()
            vis_ax.text(0.5, 0.5, 0.5, "No data to display.", ha='center', va='center', transform=vis_ax.transAxes)
            if vis_fig: vis_fig.canvas.draw_idle()
        if vis_button_prev and vis_button_prev.ax: vis_button_prev.ax.set_visible(False)
        if vis_button_next and vis_button_next.ax: vis_button_next.ax.set_visible(False)
        return

    vis_ax.clear()

    if not (0 <= vis_current_tote_idx < len(vis_tote_ids_list)):
        vis_ax.text(0.5,0.5,0.5, f"Error: Invalid tote index {vis_current_tote_idx}.", ha='center', va='center')
        if vis_fig: vis_fig.canvas.draw_idle()
        return

    current_tote_id_vis = vis_tote_ids_list[vis_current_tote_idx]
    tote_data_vis = vis_parsed_totes_data[current_tote_id_vis]
    tote_dims_vis = tote_data_vis['dims']
    items_in_tote_vis = tote_data_vis['items']

    if not vis_case_colors_for_current_tote and items_in_tote_vis:
        vis_case_colors_for_current_tote = get_distinct_colors(len(items_in_tote_vis))
    elif not items_in_tote_vis and vis_case_colors_for_current_tote:
        vis_case_colors_for_current_tote = []

    if not all(d > 0 for d in tote_dims_vis):
        vis_ax.text(0.5, 0.5, 0.5, f"Tote {current_tote_id_vis} has invalid dimensions.", ha='center', va='center', transform=vis_ax.transAxes)
        if vis_fig: vis_fig.canvas.draw_idle()
        return

    plot_cube(vis_ax, [0, 0, 0], tote_dims_vis, color='lightgrey', alpha=0.05, edges=True, face_alpha_override=0.03)

    title_str = f"Tote {current_tote_id_vis} - "
    current_step_utilization = 0.0

    if vis_current_step_in_tote == -1:
        title_str += "Step 0: Empty Tote"
    else:
        for i in range(vis_current_step_in_tote + 1):
            if i < len(items_in_tote_vis):
                item_vis = items_in_tote_vis[i]
                if not all(d > 0 for d in item_vis['dims']) or any(p is None for p in item_vis['pos']):
                    continue
                color_idx = i % len(vis_case_colors_for_current_tote) if vis_case_colors_for_current_tote else 0
                current_item_color = vis_case_colors_for_current_tote[color_idx] if vis_case_colors_for_current_tote else 'purple'
                plot_cube(vis_ax, item_vis['pos'], item_vis['dims'], color=current_item_color, alpha=0.8, label=item_vis['sku'])

        if 0 <= vis_current_step_in_tote < len(items_in_tote_vis):
            item_at_current_step = items_in_tote_vis[vis_current_step_in_tote]
            current_item_sku_vis = item_at_current_step['sku']
            current_step_utilization = item_at_current_step.get('current_utilization', 0.0)
            title_str += f"Step {vis_current_step_in_tote + 1}: Added {current_item_sku_vis}"
        else:
            title_str += f"Step {vis_current_step_in_tote + 1}"

        if vis_current_step_in_tote + 1 >= len(items_in_tote_vis):
            title_str += " (All items shown)"

    title_str += f" (Util: {current_step_utilization:.2f}%)"

    vis_ax.set_title(title_str, fontsize=10)
    vis_ax.set_xlabel("Length (X, mm)", fontsize=8)
    vis_ax.set_ylabel("Width (Y, mm)", fontsize=8)
    vis_ax.set_zlabel("Height (Z, mm)", fontsize=8)
    vis_ax.tick_params(axis='both', which='major', labelsize=7)

    vis_ax.set_xlim([0, tote_dims_vis[0]])
    vis_ax.set_ylim([0, tote_dims_vis[1]])
    vis_ax.set_zlim([0, tote_dims_vis[2]])

    try:
        vis_ax.set_box_aspect([np.ptp(vis_ax.get_xlim()), np.ptp(vis_ax.get_ylim()), np.ptp(vis_ax.get_zlim())])
    except AttributeError: pass

    vis_ax.view_init(elev=25, azim=135)
    if vis_fig: vis_fig.canvas.draw_idle()

    can_go_prev = not (vis_current_tote_idx == 0 and vis_current_step_in_tote == -1)
    max_step_current_tote_vis = len(items_in_tote_vis) - 1 if items_in_tote_vis else -1
    can_go_next = not (vis_current_tote_idx == len(vis_tote_ids_list) - 1 and vis_current_step_in_tote >= max_step_current_tote_vis)

    if vis_button_prev and vis_button_prev.ax: vis_button_prev.set_active(can_go_prev)
    if vis_button_next and vis_button_next.ax: vis_button_next.set_active(can_go_next)

def on_next_visualization(event):
    global vis_current_tote_idx, vis_current_step_in_tote, vis_case_colors_for_current_tote
    if not vis_tote_ids_list: return
    current_tote_id_vis = vis_tote_ids_list[vis_current_tote_idx]
    items_in_current_tote_vis = vis_parsed_totes_data[current_tote_id_vis]['items']
    max_step_for_current_tote_vis = len(items_in_current_tote_vis) - 1
    if vis_current_step_in_tote < max_step_for_current_tote_vis:
        vis_current_step_in_tote += 1
    else:
        if vis_current_tote_idx < len(vis_tote_ids_list) - 1:
            vis_current_tote_idx += 1
            vis_current_step_in_tote = -1
            new_tote_id_vis = vis_tote_ids_list[vis_current_tote_idx]
            vis_case_colors_for_current_tote = get_distinct_colors(len(vis_parsed_totes_data[new_tote_id_vis]['items']))
    update_visualization_display()

def on_prev_visualization(event):
    global vis_current_tote_idx, vis_current_step_in_tote, vis_case_colors_for_current_tote
    if not vis_tote_ids_list: return
    if vis_current_step_in_tote > -1:
        vis_current_step_in_tote -= 1
    else:
        if vis_current_tote_idx > 0:
            vis_current_tote_idx -= 1
            new_tote_id_vis = vis_tote_ids_list[vis_current_tote_idx]
            items_in_new_tote_vis = vis_parsed_totes_data[new_tote_id_vis]['items']
            vis_current_step_in_tote = len(items_in_new_tote_vis) - 1
            vis_case_colors_for_current_tote = get_distinct_colors(len(items_in_new_tote_vis))
    update_visualization_display()

def launch_visualization(simulation_data_for_vis):
    global vis_fig, vis_ax, vis_tote_ids_list, vis_current_tote_idx, vis_current_step_in_tote
    global vis_button_prev, vis_button_next, vis_case_colors_for_current_tote, vis_parsed_totes_data

    vis_parsed_totes_data = format_simulation_data_for_visualization(simulation_data_for_vis)
    if not vis_parsed_totes_data:
        print("No data formatted for visualization. Cannot create visualization.")
        try:
            fig_no_data, ax_no_data = plt.subplots(); ax_no_data.text(0.5, 0.5, "No tote data.", ha="center", va="center"); plt.show()
        except Exception as e: print(f"Matplotlib display error: {e}")
        return

    vis_tote_ids_list = sorted(list(vis_parsed_totes_data.keys()))
    if not vis_tote_ids_list:
        print("Tote IDs list is empty.");
        try:
            fig_no_data, ax_no_data = plt.subplots(); ax_no_data.text(0.5, 0.5, "No totes for display.", ha="center", va="center"); plt.show()
        except Exception as e: print(f"Matplotlib display error: {e}")
        return

    vis_current_tote_idx = 0
    vis_current_step_in_tote = -1
    first_tote_id_vis = vis_tote_ids_list[vis_current_tote_idx]
    vis_case_colors_for_current_tote = get_distinct_colors(len(vis_parsed_totes_data[first_tote_id_vis]['items']))

    vis_fig = plt.figure(figsize=(10, 8.5))
    vis_ax = vis_fig.add_axes([0.05, 0.15, 0.9, 0.8], projection='3d')
    ax_prev_vis = vis_fig.add_axes([0.3, 0.05, 0.15, 0.075])
    vis_button_prev = Button(ax_prev_vis, 'Previous')
    vis_button_prev.on_clicked(on_prev_visualization)
    ax_next_vis = vis_fig.add_axes([0.55, 0.05, 0.15, 0.075])
    vis_button_next = Button(ax_next_vis, 'Next')
    vis_button_next.on_clicked(on_next_visualization)
    update_visualization_display()
    plt.show()
