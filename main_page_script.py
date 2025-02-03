import streamlit as st
import os
import numpy as np
from tifffile import imread
import json
import plotly.express as px
from streamlit_plotly_events import plotly_events
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import copy    

def crop_2d_with_pad(
    image_2d: np.ndarray,
    center_row: int,
    center_col: int,
    size: int = 64
) -> np.ndarray:
    """
    Crop a 2D image around (center_row, center_col) to (size, size).
    If out of bounds, zero-pad to ensure final shape is (size, size).
    """
    half = size // 2
    r_start = center_row - half
    r_end   = center_row + half
    c_start = center_col - half
    c_end   = center_col + half

    # Valid region within image boundaries
    valid_r_start = max(r_start, 0)
    valid_r_end   = min(r_end,   image_2d.shape[0])
    valid_c_start = max(c_start, 0)
    valid_c_end   = min(c_end,   image_2d.shape[1])

    # Slice out the valid region
    sub_img = image_2d[valid_r_start:valid_r_end, valid_c_start:valid_c_end]

    # Calculate how many rows/cols we need to pad on each side
    pad_top    = valid_r_start - r_start
    pad_bottom = r_end - valid_r_end
    pad_left   = valid_c_start - c_start
    pad_right  = c_end - valid_c_end

    # Pad with zeros to get final (size, size)
    sub_img_padded = np.pad(
        sub_img,
        pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
        mode='constant',
        constant_values=0
    )

    return sub_img_padded

def encode_masks_to_rgb(mask, colormap):
    """
    Convert a uint16 mask array to an RGB image using a fixed colormap.

    :param mask: 2D numpy array (uint16) where each unique integer represents a mask.
    :param colormap: Dictionary mapping integer mask values to RGB colors (0-255).
    :return: RGB numpy array of shape (H, W, 3) with dtype uint8.
    """
    # Get image dimensions
    height, width = mask.shape

    # Create an empty RGB image
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Apply colormap
    for label, color in colormap.items():
        rgb_image[mask == label] = color

    return rgb_image

def generate_soft_colormap(n_colors=300, seed=42, mix_ratio=0.5):
    """
    Generate a colormap with `n_colors` visually distinct but softer RGB colors.
    
    :param n_colors: Number of colors to generate.
    :param seed: Random seed for reproducibility.
    :param mix_ratio: How much to blend with white (0 = full color, 1 = white).
    :return: Dictionary mapping integer mask values to RGB tuples.
    """
    np.random.seed(seed)
    
    # Generate distinct colors using HSV, then convert to RGB
    colors = plt.cm.hsv(np.linspace(0, 1, n_colors))[:, :3]  # HSV colormap
    colors = (colors * 255).astype(np.uint8)

    # Shuffle colors to ensure distinct separation
    np.random.shuffle(colors)

    # Blend each color with white to soften it
    white = np.array([255, 255, 255])
    soft_colors = (colors * (1 - mix_ratio) + white * mix_ratio).astype(np.uint8)

    return {i + 1: tuple(soft_colors[i]) for i in range(n_colors)}

def toggle_image():
    st.session_state.is_mask = not st.session_state.is_mask

# Click on information figures, show the corresponding cell in that time frame
def handle_click(clicked_fig, selected_data, selected_key, folder_path):
    if clicked_fig:
        cp = clicked_fig[0]
        clicked_time_frame = int(cp["x"])
        #st.write(f"User clicked time frame = {clicked_time_frame}")

        time_str = str(clicked_time_frame)

        if time_str in selected_data:
            centroid = selected_data[time_str].get("centroid", None)
            if centroid:
                centroid = [int(x) for x in centroid]
                tmp_file_path = os.path.join(folder_path, f"mask{clicked_time_frame:03d}t.tif")
                tmp_file = imread(tmp_file_path)
                tmp_crop = crop_2d_with_pad(
                    tmp_file[centroid[0], :, :],
                    center_row=centroid[1],
                    center_col=centroid[2],
                    size=64
                )
                st.write(f"Cell {selected_key} @ time {clicked_time_frame}: {centroid}")
                st.session_state["clicked_image"] = tmp_crop
            else:
                st.write(f"No 'centroid' found for cell {selected_key} at time {time_str}")
        else:
            st.write(f"No data found for cell {selected_key} at time {time_str}")

# Click on information figures, show the corresponding cell in that time frame
def handle_click_neighbor(clicked_tf, selected_data, selected_key, folder_path):
    if clicked_tf > 0:
        time_str = str(clicked_tf)

        if time_str in selected_data:
            centroid = selected_data[time_str].get("centroid", None)
            if centroid:
                centroid = [int(x) for x in centroid]
                tmp_file_path = os.path.join(folder_path, f"mask{clicked_tf:03d}t.tif")
                tmp_file = imread(tmp_file_path)
                tmp_crop = crop_2d_with_pad(
                    tmp_file[centroid[0], :, :],
                    center_row=centroid[1],
                    center_col=centroid[2],
                    size=64
                )
                st.write(f"Cell {selected_key} @ time {clicked_tf}: {centroid}")
                st.session_state["clicked_image_neighbor"] = tmp_crop
            else:
                st.write(f"No 'centroid' found for cell {selected_key} at time {time_str}")
        else:
            st.write(f"No data found for cell {selected_key} at time {time_str}")

# Helper function to create a figure
def create_figure(time_frames, current_time_frame, y_values, title, y_label):
    fig = go.Figure()

    # Main trace
    fig.add_trace(
        go.Scatter(
            x=time_frames,
            y=y_values,
            mode='lines+markers',
            marker=dict(color='blue'),
            line=dict(width=2),
            showlegend=False  # Hide legend
        )
    )

    # Highlight current time frame
    if current_time_frame in set(time_frames):  # Avoid index errors
        idx = time_frames.index(current_time_frame)
        fig.add_trace(
            go.Scatter(
                x=[time_frames[idx]],
                y=[y_values[idx]],
                mode='markers',
                marker=dict(color='red', size=12, symbol='circle-open'),
                showlegend=False  # Hide legend
            )
        )

    fig.update_layout(
        title=title,
        width=400,  
        xaxis_title="Time Frame",
        yaxis_title=y_label,
        #hovermode="x unified",
        #xaxis=dict(gridcolor='lightgrey', tickmode='array', tickvals=time_frames),
        #yaxis=dict(gridcolor='lightgrey'),
        showlegend=False  # Hide legend in layout
    )
    
    return fig

def main():

    # Generate colormap for masks
    distinct_colormap = generate_soft_colormap()

    # Edit page name
    st.set_page_config(
        page_title="Cell Data Extractor",
        layout="wide",
        initial_sidebar_state='collapsed'
    )

    # Initialize session state for the "click image"
    if "clicked_image_colored" not in st.session_state:
        # By default, a 64x64 zero image
        st.session_state["clicked_image_colored"] = np.zeros((3, 64, 64), dtype=np.uint8)
    if "clicked_image" not in st.session_state:
        # By default, a 64x64 zero image
        st.session_state["clicked_image"] = np.zeros((64, 64), dtype=np.uint8)
    # Initialize session state for the "click image"
    if "clicked_image_colored_neighbor" not in st.session_state:
        # By default, a 64x64 zero image
        st.session_state["clicked_image_colored_neighbor"] = np.zeros((3, 64, 64), dtype=np.uint8)
    if "clicked_image_neighbor" not in st.session_state:
        # By default, a 64x64 zero image
        st.session_state["clicked_image_neighbor"] = np.zeros((64, 64), dtype=np.uint8)
    # Initialize session state of last clicked time frame
    if "last_clicked" not in st.session_state:
        st.session_state["last_clicked"] = {key: set() for key in range(1, 9)}
    # Initialize mask / image display mode
    if "is_mask" not in st.session_state:
        st.session_state.is_mask = True
    # ---------------------------------------
    # Get json file
    # ---------------------------------------
    with st.sidebar.expander("Select JSON file", expanded=False):
        json_folder_path = st.text_input(
            "Json file of cell information",
            value="/work/scratch/zhuchen/SAM-Med3D-Updated/analyse",
            help="Enter absolute/relative path."
        )

        if not os.path.isdir(json_folder_path):
            st.warning("Invalid folder path.")
        else:
            json_files = [f for f in os.listdir(json_folder_path) if f.endswith('.json')]

            if json_files:
                selected_json_file = st.selectbox("Select general JSON file", json_files)
                # Read the selected JSON file
                file_path = os.path.join(json_folder_path, selected_json_file)
                with open(file_path, 'r') as f:
                    data = json.load(f)
            else:
                st.warning("No JSON files found in the specified folder. Please add files to the folder and reload the page.")

            if json_files:
                selected_json_file = st.selectbox("Select neighbor information JSON file", json_files)
                # Read the selected JSON file
                file_path = os.path.join(json_folder_path, selected_json_file)
                with open(file_path, 'r') as f:
                    neighbor_data = json.load(f)
            else:
                st.warning("No JSON files found in the specified folder. Please add files to the folder and reload the page.")

    # ---------------------------------------
    # Get mask file
    # ---------------------------------------
    with st.sidebar.expander("Select paths to masks and images", expanded=False):
        folder_path = st.text_input(
            "Folder containing segmentation masks:",
            value="/netshares/BiomedicalImageAnalysis/Projects/KnautNYU_PrimordiumCellSegmentation/2024_12_16_NYU4_TrackingManualCorrection/SegmentationMembranes",
            help="Enter absolute/relative path."
        )
        
        img_path = st.text_input(
            "Folder containing segmentation masks:",
            value="/netshares/BiomedicalImageAnalysis/Data/KnautNYU_PrimordiumCellSegmentation/20241127_H2A-GFP_sox10_prim_homo_40x_oil_time_frame_60/Membranes_denoised",
            help="Enter absolute/relative path."
        )

        if not os.path.isdir(folder_path):
            st.warning("Invalid folder path: Segmentation Masks.")

        if not os.path.isdir(img_path):
            st.warning("Invalid folder path: Membranes Denoised.")

    # Get tiff files
    tif_files = sorted(
        [f for f in os.listdir(folder_path) if f.lower().endswith(".tif")]
    )

    img_files = sorted(
        [f for f in os.listdir(img_path) if f.lower().endswith(".tif")]
    )

    if not tif_files:
        st.warning("No .tif files found in the specified folder.")
        return

    if not img_files:
        st.warning("No .tif files found in the specified folder.")
        return

    # ---------------------------------------
    # Define first row
    # ---------------------------------------
    col1, col2, col3, col4 = st.columns([2, 6, 1, 1])  # Adjust column width ratio if needed

    # Time frame selection
    with col1:
        selected_file = st.selectbox("Select time frame", tif_files)

    file_path = os.path.join(folder_path, selected_file)

    #st.write(f"Reading file: `{selected_file}` ...")

    current_time_frame = int(selected_file[4:7])  # Assumes consistent filename format

    selected_img_path = os.path.join(img_path, img_files[current_time_frame])

    # Read the data (Z, Y, X)
    volume_data = imread(file_path)
    img_volume_data = imread(selected_img_path)
    
    z_size, height_y, width_x = volume_data.shape

    # Get the z slice in image
    with col2:
        z_index = st.slider(
            "Select Z slice",
            min_value=0,
            max_value=z_size - 1,
            value=0
        )

    # Set crop window size
    with col3:
        with st.popover("Window size"):
            input_value = st.number_input("Enter window size", value=64, step=1)

    # Button for mask / image mode change
    with col4:
        st.button(
            "Raw Image" if st.session_state.is_mask else "Masks",
            on_click=toggle_image
        )

    # ---------------------------------------
    # Main visualization window
    # ---------------------------------------
    # Get the slice and change to a colored image
    selected_slice = volume_data[z_index, :, :]  # shape: (Y, X)
    colored_slice = encode_masks_to_rgb(selected_slice, colormap=distinct_colormap)

    # Define display dimensions for main figure
    display_width = 1300
    display_height = 300

    # Display mask or image
    if st.session_state.is_mask: 
        fig_main = px.imshow(
            colored_slice,
            #color_continuous_scale='gray',
            #origin='upper',
            title=f"Slice Z={z_index}",
        )
        # Change information showed when mouse on image
        fig_main.update_traces(
            customdata=selected_slice,  # Store original values
            hovertemplate="Mask Value: %{customdata}<extra></extra>",  # Display raw mask value
        )
    else:
        fig_main = px.imshow(
            img_volume_data[z_index, :, :],
            color_continuous_scale='gray',
            #origin='upper',
            title=f"Slice Z={z_index}",
        )

    fig_main.update_layout(
        width=display_width,
        height=display_height,
        coloraxis_showscale=False,
        margin=dict(l=0, r=0, t=30, b=0),
        dragmode='pan'
    )

    clicked_points_main = plotly_events(
        fig_main,
        click_event=True,
        hover_event=False,
        select_event=False,
        override_height= display_height,
        override_width=display_width,
        key="main_fig_select"
    )
    
    # ---------------------------------------
    # Click process
    # ---------------------------------------
    pixel_val = 0
    if clicked_points_main:
        cp = clicked_points_main[0]
        x_coord = int(cp["x"])
        y_coord = int(cp["y"])

        if 0 <= x_coord < width_x and 0 <= y_coord < height_y:
            pixel_val = selected_slice[y_coord, x_coord]
            st.write(f"**Clicked coordinates:** (Z={z_index}, Y={y_coord}, X={x_coord})     **Selected Cell Index:** {pixel_val}")

            if st.session_state.is_mask: 
                xy_slice = volume_data[z_index, :, :]
                xz_slice = volume_data[:, y_coord, :]
                yz_slice = volume_data[:, :, x_coord]
            else:
                xy_slice = img_volume_data[z_index, :, :]
                xz_slice = img_volume_data[:, y_coord, :]
                yz_slice = img_volume_data[:, :, x_coord]

            # Crop patch in three dimensions
            xy_crop = crop_2d_with_pad(
                xy_slice,
                center_row=y_coord,
                center_col=x_coord,
                size=input_value
            )

            xz_crop = crop_2d_with_pad(
                xz_slice,
                center_row=z_index,
                center_col=x_coord,
                size=input_value
            )

            yz_crop = crop_2d_with_pad(
                yz_slice,
                center_row=z_index,
                center_col=y_coord,
                size=input_value
            )

            # Define three columns
            col1, col2, col3 = st.columns(3)
            display_width_patch = 240
            
            if st.session_state.is_mask: 
                colored_xy_slice = encode_masks_to_rgb(xy_crop, colormap=distinct_colormap)
                fig_xy = px.imshow(
                    colored_xy_slice,
                    #color_continuous_scale='gray',
                    origin='upper',
                    title="XY plane"
                )
                fig_xy.update_traces(
                    customdata=xy_crop,  # Store original values
                    hovertemplate="Mask Value: %{customdata}<extra></extra>",  # Display raw mask value
                )
                fig_xy.update_layout(
                    width = display_width_patch,
                    height = display_width_patch,
                    margin=dict(l=0, r=0, t=30, b=0),
                    coloraxis_showscale=False
                )
                col1.plotly_chart(fig_xy, use_container_width=False)

                colored_xz_slice = encode_masks_to_rgb(xz_crop, colormap=distinct_colormap)
                fig_xz = px.imshow(
                    colored_xz_slice,
                    #color_continuous_scale='gray',
                    origin='upper',
                    title="XZ plane"
                )
                fig_xz.update_traces(
                    customdata=xz_crop,  # Store original values
                    hovertemplate="Mask Value: %{customdata}<extra></extra>",  # Display raw mask value
                )
                fig_xz.update_layout(
                    width = display_width_patch,
                    height = display_width_patch,
                    margin=dict(l=0, r=0, t=30, b=0),
                    coloraxis_showscale=False
                )
                col2.plotly_chart(fig_xz, use_container_width=False)

                colored_yz_slice = encode_masks_to_rgb(yz_crop, colormap=distinct_colormap)
                fig_yz = px.imshow(
                    colored_yz_slice,
                    #color_continuous_scale='gray',
                    origin='upper',
                    title="YZ plane"
                )
                fig_yz.update_traces(
                    customdata=yz_crop,  # Store original values
                    hovertemplate="Mask Value: %{customdata}<extra></extra>",  # Display raw mask value
                )
                fig_yz.update_layout(
                    width = display_width_patch,
                    height = display_width_patch,
                    margin=dict(l=0, r=0, t=30, b=0),
                    coloraxis_showscale=False
                )
                col3.plotly_chart(fig_yz, use_container_width=False)
            else:
                fig_xy = px.imshow(
                    xy_crop,
                    color_continuous_scale='gray',
                    origin='upper',
                    title="XY plane (z fixed)"
                )
                fig_xy.update_layout(
                    width = display_width_patch,
                    height = display_width_patch,
                    margin=dict(l=0, r=0, t=30, b=0),
                    coloraxis_showscale=False
                )
                col1.plotly_chart(fig_xy, use_container_width=False)

                fig_xz = px.imshow(
                    xz_crop,
                    color_continuous_scale='gray',
                    origin='upper',
                    title="XZ plane (y fixed)"
                )
                fig_xz.update_layout(
                    width = display_width_patch,
                    height = display_width_patch,
                    margin=dict(l=0, r=0, t=30, b=0),
                    coloraxis_showscale=False
                )
                col2.plotly_chart(fig_xz, use_container_width=False)

                fig_yz = px.imshow(
                    yz_crop,
                    color_continuous_scale='gray',
                    origin='upper',
                    title="YZ plane (x fixed)"
                )
                fig_yz.update_layout(
                    width = display_width_patch,
                    height = display_width_patch,
                    margin=dict(l=0, r=0, t=30, b=0),
                    coloraxis_showscale=False
                )
                col3.plotly_chart(fig_yz, use_container_width=False)

    # ---------------------------------------
    # Display the cell information
    # ---------------------------------------
    if selected_file:
        if pixel_val != 0:
            # Convert int to string for JSON lookup
            selected_key = str(int(pixel_val))
            if selected_key not in data:
                st.warning(f"No data found in the JSON for cell ID = {selected_key}")
                return
            selected_data = data[selected_key]

            # Extract time frame indices and corresponding data
            time_frames = []
            surface_areas = []
            max_lengths_x = []
            max_lengths_y = []
            max_lengths_z = []
            volumes = []
            movements = []
            num_neighbors = []
            changed_neighbors = []
            #connections = {}

            for tf_name, mask_data in selected_data.items():
                # Extract the time frame index from the filename
                time_frame = int(tf_name)
                time_frames.append(time_frame)

                # Extract the required values
                surface_areas.append(mask_data.get("surface_area", 0))
                max_lengths_x.append(mask_data.get("max_length_x", 0))
                max_lengths_y.append(mask_data.get("max_length_y", 0))
                max_lengths_z.append(mask_data.get("max_length_z", 0))
                volumes.append(mask_data.get("volume", 0))
                movements.append(mask_data.get("movement", 0))
                num_neighbors.append(mask_data.get("num_neighbors", 0))
                changed_neighbors.append(mask_data.get("changed_neighbors", 0))
                #adjacency = mask_data.get("adjacency", 0)
                #for adj_key, adj_value in adjacency:
                #    if int(adj_key) not in connections.keys():
                #        connections[int(adj_key)] = {}
                #        connections[int]

            # Sort the data by time frame
            sorted_indices = sorted(range(len(time_frames)), key=lambda i: time_frames[i])
            time_frames = [time_frames[i] for i in sorted_indices]
            surface_areas = [surface_areas[i] for i in sorted_indices]
            max_lengths_x = [max_lengths_x[i] for i in sorted_indices]
            max_lengths_y = [max_lengths_y[i] for i in sorted_indices]
            max_lengths_z = [max_lengths_z[i] for i in sorted_indices]
            volumes = [volumes[i] for i in sorted_indices]
            movements = [movements[i] for i in sorted_indices]
            num_neighbors = [num_neighbors[i] for i in sorted_indices]
            changed_neighbors = [changed_neighbors[i] for i in sorted_indices]

            fig1 = create_figure(time_frames, current_time_frame, num_neighbors, "Number of Neighbors", "Number of Neighbors")
            fig2 = create_figure(time_frames, current_time_frame, movements, "Movement", "Movement")

            fig3 = create_figure(time_frames, current_time_frame, surface_areas, "Surface Area", "Surface Area")
            fig4 = create_figure(time_frames, current_time_frame, volumes, "Volume", "Volume")
            fig5 = create_figure(time_frames, current_time_frame, changed_neighbors, "Changed Neighbors", "Changed Neighbors")
            fig6 = create_figure(time_frames, current_time_frame, max_lengths_x, "Max Length X", "Max Length X")
            fig7 = create_figure(time_frames, current_time_frame, max_lengths_y, "Max Length Y", "Max Length Y")
            fig8 = create_figure(time_frames, current_time_frame, max_lengths_z, "Max Length Z", "Max Length Z")

            neighbors_figure_list = []
            current_cell_neighbors = neighbor_data[selected_key]
            for current_neighbor, current_neighbor_info in current_cell_neighbors.items():
                #print(current_neighbor_info)
                neighbors_figure_list.append(create_figure(current_neighbor_info["time"], current_time_frame, current_neighbor_info["size"], f"{selected_key} and {current_neighbor}", "Surface Area"))
                

            last_condition = copy.deepcopy(st.session_state["last_clicked"])
            
            with st.expander("Show more information"):
                e1, e2, e3 = st.columns(3)
                
                with e1:
                    clicked_fig3 = plotly_events(fig3, click_event=True, override_width="100%", override_height=400)
                    if clicked_fig3:
                        cp = clicked_fig3[0]
                        clicked_time_frame = int(cp["x"])
                        st.session_state["last_clicked"][3].add(clicked_time_frame)
                
                with e2:
                    clicked_fig4 = plotly_events(fig4, click_event=True, override_width="100%", override_height=400)
                    if clicked_fig4:
                        cp = clicked_fig4[0]
                        clicked_time_frame = int(cp["x"])
                        st.session_state["last_clicked"][4].add(clicked_time_frame)
                
                with e3:
                    clicked_fig5 = plotly_events(fig5, click_event=True, override_width="100%", override_height=400)
                    if clicked_fig5:
                        cp = clicked_fig5[0]
                        clicked_time_frame = int(cp["x"])
                        st.session_state["last_clicked"][5].add(clicked_time_frame)
                
                # Additional row for more images if needed
                e4, e5, e6 = st.columns(3)
                
                with e4:
                    clicked_fig6 = plotly_events(fig6, click_event=True, override_width="100%", override_height=400)
                    if clicked_fig6:
                        cp = clicked_fig6[0]
                        clicked_time_frame = int(cp["x"])
                        st.session_state["last_clicked"][6].add(clicked_time_frame)
                
                with e5:
                    clicked_fig7 = plotly_events(fig7, click_event=True, override_width="100%", override_height=400)
                    if clicked_fig7:
                        cp = clicked_fig7[0]
                        clicked_time_frame = int(cp["x"])
                        st.session_state["last_clicked"][7].add(clicked_time_frame)

                with e6:
                    clicked_fig8 = plotly_events(fig8, click_event=True, override_width="100%", override_height=400)
                    if clicked_fig8:
                        cp = clicked_fig8[0]
                        clicked_time_frame = int(cp["x"])
                        st.session_state["last_clicked"][8].add(clicked_time_frame)
            

                # Layout the 8 figures in columns
                c1, c2, c3 = st.columns(3)

                with c1:
                    clicked_fig1 = plotly_events(fig1, click_event=True, override_width="100%", override_height=400)
                    if clicked_fig1:
                        cp = clicked_fig1[0]
                        clicked_time_frame = int(cp["x"])
                        st.session_state["last_clicked"][1].add(clicked_time_frame)        
    
                with c2:
                    clicked_fig2 = plotly_events(fig2, click_event=True, override_width="100%", override_height=400)
                    if clicked_fig2:
                        cp = clicked_fig2[0]
                        clicked_time_frame = int(cp["x"])
                        st.session_state["last_clicked"][2].add(clicked_time_frame)    

                with c3:
                    # Find clicked time frame and show image
                    changed_key = {key for key in last_condition if last_condition[key] != st.session_state["last_clicked"][key]}
                    if not changed_key:
                        pass
                    elif next(iter(changed_key)) == 1:
                        handle_click(clicked_fig1, selected_data, selected_key, folder_path)
                    elif next(iter(changed_key)) == 2:
                        handle_click(clicked_fig2, selected_data, selected_key, folder_path)
                    elif next(iter(changed_key)) == 3:
                        handle_click(clicked_fig3, selected_data, selected_key, folder_path)
                    elif next(iter(changed_key)) == 4:
                        handle_click(clicked_fig4, selected_data, selected_key, folder_path)
                    elif next(iter(changed_key)) == 5:
                        handle_click(clicked_fig5, selected_data, selected_key, folder_path)
                    elif next(iter(changed_key)) == 6:
                        handle_click(clicked_fig6, selected_data, selected_key, folder_path)
                    elif next(iter(changed_key)) == 7:
                        handle_click(clicked_fig7, selected_data, selected_key, folder_path)
                    elif next(iter(changed_key)) == 8:
                        handle_click(clicked_fig8, selected_data, selected_key, folder_path)

                    st.session_state["clicked_image_colored"] = encode_masks_to_rgb(st.session_state["clicked_image"], distinct_colormap)
                    
                    fig_neighbor = px.imshow(
                        st.session_state["clicked_image_colored"],
                    )
                    fig_neighbor.update_traces(
                        customdata=st.session_state["clicked_image"],  # Store original values
                        hovertemplate="Mask Value: %{customdata}<extra></extra>",  # Display raw mask value
                    )
                    fig_neighbor.update_layout(
                        width = 400,
                        height = 400,
                        margin=dict(l=0, r=0, t=0, b=0),
                        coloraxis_showscale=False
                    )
                    clicked_points_neighbor = plotly_events(
                        fig_neighbor,
                        click_event=True,
                        hover_event=False,
                        select_event=False,
                        override_height= 400,
                        override_width=400,
                        key="neighbor_fig_select"
                    )
            cols_per_row = 3  

            with st.expander("Show more information"):
                # Dynamically create columns based on the figure count
                num_figs = len(neighbors_figure_list)
                rows = (num_figs + cols_per_row - 1) // cols_per_row  # Compute needed rows
                if "clicked_neighbor" in st.session_state:
                    last_neighbor_condition = copy.deepcopy(st.session_state["clicked_neighbor"])
                else:
                    last_neighbor_condition = set()

                for i in range(rows):
                    cols = st.columns(min(cols_per_row, num_figs - i * cols_per_row))  # Adjust for last row
                    for j, col in enumerate(cols):
                        idx = i * cols_per_row + j
                        if idx < num_figs:
                            with col:
                                clicked_fig = plotly_events(
                                    neighbors_figure_list[idx], 
                                    click_event=True, 
                                    override_width="100%", 
                                    override_height=400
                                )
                            if clicked_fig:
                                if "clicked_neighbor" not in st.session_state:
                                    # By default, a 64x64 zero image
                                    st.session_state["clicked_neighbor"] = set()
                                    #print(clicked_fig)
                                st.session_state["clicked_neighbor"].add(clicked_fig[0]['x'])

            if "clicked_neighbor" in st.session_state:                    
                for element in st.session_state["clicked_neighbor"]:
                    if element not in last_neighbor_condition:
                        handle_click_neighbor(element, selected_data, selected_key, folder_path)

                st.session_state["clicked_image_colored_neighbor"] = encode_masks_to_rgb(st.session_state["clicked_image_neighbor"], distinct_colormap)

                fig_neighbor_connection = px.imshow(
                    st.session_state["clicked_image_colored_neighbor"],
                )
                fig_neighbor_connection.update_traces(
                    customdata=st.session_state["clicked_image_neighbor"],  # Store original values
                    hovertemplate="Mask Value: %{customdata}<extra></extra>",  # Display raw mask value
                )
                fig_neighbor_connection.update_layout(
                    width = 400,
                    height = 400,
                    margin=dict(l=0, r=0, t=0, b=0),
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig_neighbor_connection)

        else:
            st.warning("Clicked outside the data range.")
    else:
        st.info("No click yet.")

if __name__ == "__main__":
    main()
