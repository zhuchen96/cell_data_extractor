import streamlit as st
import os
import numpy as np
import json
import plotly.express as px
from streamlit_plotly_events import plotly_events
import plotly.graph_objects as go
import matplotlib.pyplot as plt    
import zarr
from skimage import exposure
import re

def gamma_correction(image, gamma=0.00001):
    # Normalize image to [0,1]
    norm_image = image / image.max()
    # Apply gamma correction
    gamma_corrected = np.power(norm_image, gamma)
    # Scale back to original range if necessary
    return gamma_corrected

def contrast_stretch(image):
    image = exposure.equalize_adapthist(image, clip_limit=0.02)
    return image

def adjust_clicked_tf_cellinfo(offset):
    if "clicked_tf_cellinfo" in st.session_state and st.session_state["clicked_tf_cellinfo"] is not None:
        st.session_state["clicked_tf_cellinfo"] += offset
        st.rerun()

def adjust_clicked_tf_msrd(offset):
    if "clicked_tf_msrd" in st.session_state and st.session_state["clicked_tf_msrd"] is not None:
        st.session_state["clicked_tf_msrd"] += offset
        st.rerun()

def crop_3d_with_pad(
    image_3d,
    center_row: int,
    center_col: int,
    center_dep: int,
    size: int = 64
) -> np.ndarray:
    """
    Crop a 2D image around (center_row, center_col) to (size, size).
    If out of bounds, zero-pad to ensure final shape is (size, size).
    """
    half = size // 2
    d_start = center_dep - half // 2
    d_end = center_dep + half //2
    r_start = center_row - half
    r_end   = center_row + half
    c_start = center_col - half
    c_end   = center_col + half

    # Valid region within image boundaries
    valid_d_start = max (d_start, 0)
    valid_d_end   = min(d_end, image_3d.shape[0])
    valid_r_start = max(r_start, 0)
    valid_r_end   = min(r_end,   image_3d.shape[1])
    valid_c_start = max(c_start, 0)
    valid_c_end   = min(c_end,   image_3d.shape[2])

    # Slice out the valid region
    sub_img = image_3d[valid_d_start:valid_d_end, valid_r_start:valid_r_end, valid_c_start:valid_c_end]

    # Calculate how many rows/cols we need to pad on each side
    pad_top    = valid_r_start - r_start
    pad_bottom = r_end - valid_r_end
    pad_left   = valid_c_start - c_start
    pad_right  = c_end - valid_c_end
    pad_front   = valid_d_start - d_start
    pad_back  = d_end - valid_d_end
    # Pad with zeros to get final (size, size)
    sub_img_padded = np.pad(
        sub_img,
        pad_width=((pad_front, pad_back), (pad_top, pad_bottom), (pad_left, pad_right)),
        mode='constant',
        constant_values=np.min(image_3d)
    )

    return sub_img_padded

def crop_2d_with_pad(
    image_2d,
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
        constant_values=np.min(image_2d)
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
def handle_click(clicked_tf_neighbor, selected_data, selected_key, mask_path, image_path, input_value):
    time_str = str(clicked_tf_neighbor)

    if time_str in selected_data:
        centroid = selected_data[time_str].get("centroid", None)
        if centroid:
            centroid = [int(x) for x in centroid]
            tmp_mask_path = os.path.join(mask_path, f"t{clicked_tf_neighbor:03d}")
            tmp_img_path = os.path.join(image_path, f"t{clicked_tf_neighbor:03d}")

            #tmp_file = imread(tmp_file_path)
            tmp_mask_file = zarr.open(tmp_mask_path, mode='r')
            tmp_img_file = zarr.open(tmp_img_path, mode='r')            
            tmp_crop_mask = crop_3d_with_pad(
                tmp_mask_file,
                center_row=centroid[1],
                center_col=centroid[2],
                center_dep=centroid[0],
                size=input_value
            )
            tmp_crop_img = crop_3d_with_pad(
                tmp_img_file,
                center_row=centroid[1],
                center_col=centroid[2],
                center_dep=centroid[0],
                size=input_value
            )
            st.write(f"Cell {selected_key} @ time {clicked_tf_neighbor}: {centroid}")
            st.session_state["clicked_mask"] = tmp_crop_mask
            st.session_state["clicked_image"] = tmp_crop_img
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

def handle_click_msrd(clicked_tf_msrd, selected_data, selected_key, clicked_neighbor, mask_path, image_path, input_value):
    time_str = str(clicked_tf_msrd)

    if time_str in selected_data:
        centroid = selected_data[time_str].get("centroid", None)
        if centroid:
            centroid = [int(x) for x in centroid]
            tmp_mask_path = os.path.join(mask_path, f"t{clicked_tf_msrd:03d}")
            tmp_img_path = os.path.join(image_path, f"t{clicked_tf_msrd:03d}")

            #tmp_file = imread(tmp_file_path)
            tmp_mask_file = zarr.open(tmp_mask_path, mode='r')
            tmp_img_file = zarr.open(tmp_img_path, mode='r')
            tmp_mask_file = tmp_mask_file[:] 
            '''            
            tmp_crop_mask = crop_3d_with_pad(
                tmp_mask_file,
                center_row=centroid[1],
                center_col=centroid[2],
                center_dep=centroid[0],
                size=input_value
            )
            tmp_crop_img = crop_3d_with_pad(
                tmp_img_file,
                center_row=centroid[1],
                center_col=centroid[2],
                center_dep=centroid[0],
                size=input_value
            )
            '''

            coords = np.argwhere((tmp_mask_file == int(selected_key)) | (tmp_mask_file == int(clicked_neighbor)))

            z_min, y_min, x_min = coords.min(axis=0)
            z_max, y_max, x_max = coords.max(axis=0) + 1  # +1 because slicing is exclusive on the upper bound

            # Crop
            tmp_crop_mask = tmp_mask_file[z_min:z_max, y_min:y_max, x_min:x_max]
            tmp_crop_img = tmp_img_file[z_min:z_max, y_min:y_max, x_min:x_max]

            st.write(f"MSRD between cell {selected_key} and cell {clicked_neighbor}")
            st.session_state["clicked_mask_msrd"] = tmp_crop_mask
            st.session_state["clicked_image_msrd"] = tmp_crop_img
        else:
            st.write(f"No 'centroid' found for cell {selected_key} at time {time_str}")
    else:
        st.write(f"No data found for cell {selected_key} at time {time_str}")

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
    if "clicked_mask_colored" not in st.session_state:
        st.session_state["clicked_mask_colored"] = None
    if "clicked_mask" not in st.session_state:
        st.session_state["clicked_mask"] = None
    if "clicked_image" not in st.session_state:
        st.session_state["clicked_image"] = None


    if "clicked_mask_colored_msrd" not in st.session_state:
        st.session_state["clicked_mask_colored_msrd"] = None
    if "clicked_mask_msrd" not in st.session_state:
        st.session_state["clicked_mask_msrd"] = None
    if "clicked_image_msrdr" not in st.session_state:
        st.session_state["clicked_image_msrd"] = None

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
            value="json_files_nuc",
            help="Enter absolute/relative path."
        )

        if not os.path.isdir(json_folder_path):
            st.warning("Invalid folder path.")
        else:
            json_files = [f for f in os.listdir(json_folder_path) if f.endswith('.json')]

            if json_files:
                # Set default selection to the first file if available
                selected_json_file = st.selectbox(
                    "Select general JSON file", json_files, index=0
                )
                # Read the selected JSON file
                file_path = os.path.join(json_folder_path, selected_json_file)
                with open(file_path, 'r') as f:
                    data_load = json.load(f)
                data = data_load["cell_information"]
                msrd_data = data_load["neighbor_centroid_distances"]
            else:
                st.warning("No JSON files found in the specified folder. Please add files to the folder and reload the page.")
                data = None  # Set data to None if no files are available

    # ---------------------------------------
    # Get mask file
    # ---------------------------------------
    with st.sidebar.expander("Select paths to masks and images", expanded=False):
        folder_path = st.text_input(
            "Folder containing segmentation masks:",
            value="zarr_masks_nuc",
            help="Enter absolute/relative path."
        )
        
        img_path = st.text_input(
            "Folder containing raw images:",
            value="zarr_images_nuc",
            help="Enter absolute/relative path."
        )

        if not os.path.isdir(folder_path):
            st.warning("Invalid folder path: Segmentation Masks.")

        if not os.path.isdir(img_path):
            st.warning("Invalid folder path: Membranes Denoised.")


    # Set crop window size
    with st.sidebar.expander("Window size", expanded=False):
        with st.popover("Change"):
            input_value = st.number_input("Enter window size", value=64, step=1)

    # Button for mask / image mode change
    with st.sidebar.expander("Switch display mode", expanded=False):
        st.button(
            "Raw Image" if st.session_state.is_mask else "Masks",
            on_click=toggle_image
        )

    # Get tiff files
    tif_files = sorted(
        [f for f in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, f)) and re.match(r"^t\d{3}$", f)]
    )
    img_files = sorted(
        [f for f in os.listdir(img_path)
        if os.path.isdir(os.path.join(img_path, f)) and re.match(r"^t\d{3}$", f)]
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
    col1, col2 = st.columns([2, 8])  # Adjust column width ratio if needed

    # Time frame selection
    with col1:
        selected_file = st.selectbox("Select time frame", tif_files)

    file_path = os.path.join(folder_path, selected_file)

    #st.write(f"Reading file: `{selected_file}` ...")

    current_time_frame = int(selected_file[1:4])  # Assumes consistent filename format

    selected_img_path = os.path.join(img_path, img_files[current_time_frame])

    # Read the data (Z, Y, X)
    #volume_data = imread(file_path)
    #img_volume_data = imread(selected_img_path)
    volume_data = zarr.open(file_path, mode='r')
    img_volume_data = zarr.open(selected_img_path, mode='r')


    z_size, height_y, width_x = volume_data.shape

    # Get the z slice in image
    with col2:
        z_index = st.slider(
            "Select Z slice",
            min_value=0,
            max_value=z_size - 1,
            value=0
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
            customdata=selected_slice.astype(str),  # Store original values
            hovertemplate="Mask Value: %{customdata}<extra></extra>",  # Display raw mask value
        )
    else:
        fig_main = px.imshow(
            contrast_stretch(img_volume_data[z_index, :, :]),
            binary_string=True,
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
                    customdata=xy_crop.astype(str),  # Store original values
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
                    customdata=xz_crop.astype(str),  # Store original values
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
                    customdata=yz_crop.astype(str),  # Store original values
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
                    contrast_stretch(xy_crop),
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
                    contrast_stretch(xz_crop),
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
                    contrast_stretch(yz_crop),
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

            # Sort the data by time frame
            sorted_indices = sorted(range(len(time_frames)), key=lambda i: time_frames[i])
            time_frames = [time_frames[i] for i in sorted_indices]
            surface_areas = [surface_areas[i] for i in sorted_indices]
            max_lengths_x = [max_lengths_x[i] for i in sorted_indices]
            max_lengths_y = [max_lengths_y[i] for i in sorted_indices]
            max_lengths_z = [max_lengths_z[i] for i in sorted_indices]
            volumes = [volumes[i] for i in sorted_indices]
            movements = [movements[i] for i in sorted_indices]

            fig2 = create_figure(time_frames, current_time_frame, movements, "Movement", "Movement")
            fig3 = create_figure(time_frames, current_time_frame, surface_areas, "Surface Area", "Surface Area")
            fig4 = create_figure(time_frames, current_time_frame, volumes, "Volume", "Volume")
            fig6 = create_figure(time_frames, current_time_frame, max_lengths_x, "Max Length X", "Max Length X")
            fig7 = create_figure(time_frames, current_time_frame, max_lengths_y, "Max Length Y", "Max Length Y")
            fig8 = create_figure(time_frames, current_time_frame, max_lengths_z, "Max Length Z", "Max Length Z")

            cellinfo_figure_list = [fig2, fig3, fig4, fig6, fig7, fig8]

            if "click_dict_cellinfo" not in st.session_state:
                st.session_state["click_dict_cellinfo"] = {}
            
            cols_per_row = 3
            with st.expander("Show more information"):
                num_figs = len(cellinfo_figure_list)
                rows = (num_figs + cols_per_row - 1) // cols_per_row  # Compute needed rows
                if "clicked_tf_cellinfo" not in st.session_state:
                    st.session_state["clicked_tf_cellinfo"] = None
                for i in range(rows):
                    cols = st.columns(min(cols_per_row, num_figs - i * cols_per_row))  # Adjust for last row
                    for j, col in enumerate(cols):
                        idx = i * cols_per_row + j
                        if idx < num_figs:
                            with col:
                                # Plot figure and allow click
                                clicked_cellinfo_fig = plotly_events(
                                    cellinfo_figure_list[idx], 
                                    click_event=True, 
                                    override_width="100%", 
                                    override_height=400
                                )
                            if clicked_cellinfo_fig:
                                # Get click, if it's the first click of a figure, generate a new element
                                if idx not in st.session_state["click_dict_cellinfo"].keys():
                                    st.session_state["click_dict_cellinfo"][idx] = clicked_cellinfo_fig
                                    # Get the clicked time frame 
                                    st.session_state["clicked_tf_cellinfo"] = clicked_cellinfo_fig[0]['x']
                                else:
                                    # Check if the current click is the same as last click
                                    if st.session_state["click_dict_cellinfo"][idx] != clicked_cellinfo_fig:
                                        # Get clicked time frame if the current click is different (it's a real click)
                                        st.session_state["clicked_tf_cellinfo"] = clicked_cellinfo_fig[0]['x']
                                        st.session_state["click_dict_cellinfo"][idx] = clicked_cellinfo_fig
                                    else:
                                        pass
                if st.session_state["clicked_tf_cellinfo"] is not None:
                    handle_click(st.session_state["clicked_tf_cellinfo"], selected_data, selected_key, folder_path, img_path, input_value)
                                        
                    z_index_cellinfo = st.slider(
                        "Select Z slice",
                        min_value=0,
                        max_value=input_value//2 - 1,
                        value= input_value//4,
                        key="cell_info_slider"
                    )
                    if st.session_state.is_mask: 
                        cell_mask = st.session_state["clicked_mask"][z_index_cellinfo, :, :]
                        cell_mask_colored = encode_masks_to_rgb(cell_mask, distinct_colormap)
                    else:
                        cell_img = st.session_state["clicked_image"][z_index_cellinfo, :, :]

                    col1, col2, col3 = st.columns([1, 6, 1])  # Create layout for buttons
                    
                    with col1:
                        if st.button("←", key="prev_button"):
                            adjust_clicked_tf_cellinfo(-1)
                    with col2:
                        if st.session_state.is_mask: 
                            fig_neighbor = px.imshow(
                                cell_mask_colored,
                            )
                            fig_neighbor.update_traces(
                            customdata=cell_mask.astype(str),  # Store original values
                            hovertemplate="Mask Value: %{customdata}<extra></extra>",  # Display raw mask value
                            )
                        else:
                            fig_neighbor = px.imshow(
                                contrast_stretch(cell_img),
                                binary_string=True
                            )                                        

                        st.plotly_chart(fig_neighbor, key="neighbor_fig")        
                     
                    with col3:
                        if st.button("→", key="next_button"):
                            adjust_clicked_tf_cellinfo(1)  

            # -------------------------------------------------Show MSRD Information------------------------------------------------------------
            if "click_dict_msrd" not in st.session_state:
                st.session_state["click_dict_msrd"] = {}

            with st.expander("Show MSRD information"):
                if selected_key not in msrd_data:
                    st.warning(f"No data found in the JSON for cell ID = {selected_key}")
                    return

                msrd_figure_list = []
                msrd_index_list = []

                for neighbor_id, dist_list in msrd_data[selected_key].items():
                    distances = [(d-dist_list[0])**2/25**2 if d is not None else float('nan') for d in dist_list]
                    msrd_figure_list.append(
                        create_figure(time_frames, current_time_frame, distances, 
                                    f"MSRD {selected_key} - {neighbor_id}", 
                                    f"MSRD {selected_key} - {neighbor_id}")
                    )
                    msrd_index_list.append(neighbor_id)

                num_figs = len(msrd_index_list)
                rows = (num_figs + cols_per_row - 1) // cols_per_row

                # Session state initialization
                if "clicked_tf_msrd" not in st.session_state:
                    st.session_state["clicked_tf_msrd"] = None
                if "clicked_neighbor" not in st.session_state:
                    st.session_state["clicked_neighbor"] = 0
                if "click_dict_msrd" not in st.session_state:
                    st.session_state["click_dict_msrd"] = {}

                # Display each row under a toggle (checkbox)
                for i in range(rows):
                    show_row = st.checkbox(f"Show Row {i+1}", value=True)
                    if show_row:
                        cols = st.columns(min(cols_per_row, num_figs - i * cols_per_row))
                        for j, col in enumerate(cols):
                            idx = i * cols_per_row + j
                            if idx < num_figs:
                                with col:
                                    clicked_fig_msrd = plotly_events(
                                        msrd_figure_list[idx], 
                                        click_event=True, 
                                        override_width="100%", 
                                        override_height=400
                                    )

                                if clicked_fig_msrd:
                                    prev_click = st.session_state["click_dict_msrd"].get(idx)
                                    if prev_click != clicked_fig_msrd:
                                        st.session_state["click_dict_msrd"][idx] = clicked_fig_msrd
                                        st.session_state["clicked_tf_msrd"] = clicked_fig_msrd[0]['x']
                                        st.session_state["clicked_neighbor"] = msrd_index_list[idx]

                if st.session_state["clicked_tf_msrd"] is not None:
                    handle_click_msrd(st.session_state["clicked_tf_msrd"], selected_data, selected_key, st.session_state["clicked_neighbor"], folder_path, img_path, input_value)
                                        
                    z_index_cellinfo = st.slider(
                        "Select Z slice",
                        min_value=0,
                        max_value=st.session_state["clicked_mask_msrd"].shape[0] - 1,
                        value= input_value//4,
                        key="cell_neighbor_slider"
                    )
                    if st.session_state.is_mask: 
                        cell_neighbor_mask = st.session_state["clicked_mask_msrd"][z_index_cellinfo, :, :]
                        cell_neighbor_mask_colored = encode_masks_to_rgb(cell_neighbor_mask, distinct_colormap)
                    else:
                        cell_neighbor_img = st.session_state["clicked_image_msrd"][z_index_cellinfo, :, :]

                    col1, col2, col3 = st.columns([1, 6, 1])  # Create layout for buttons
                    
                    with col1:
                        if st.button("←", key="prev_button_msrd"):
                            adjust_clicked_tf_msrd(-1)
                    
                    with col2:
                        if st.session_state.is_mask:
                            fig_neighbor_connection = px.imshow(cell_neighbor_mask_colored)

                            fig_neighbor_connection.update_traces(
                            customdata=cell_neighbor_mask.astype(str),  # Store original values
                            hovertemplate="Mask Value: %{customdata}<extra></extra>",  # Display raw mask value
                            )
                        else:
                            fig_neighbor_connection = px.imshow(contrast_stretch(cell_neighbor_img), binary_string=True)
                        

                        st.plotly_chart(fig_neighbor_connection, key="_msrd_connection_fig")
                    
                    with col3:
                        if st.button("→", key="next_button_msrd"):
                            adjust_clicked_tf_msrd(1)


        else:
            st.warning("Clicked outside the data range.")
    else:
        st.info("No click yet.")

if __name__ == "__main__":
    main()
