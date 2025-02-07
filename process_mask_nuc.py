import os
import zarr
import json
import numpy as np
import argparse
import tifffile as tiff
from collections import defaultdict

def gather_label_info(data):
    label_info = defaultdict(lambda: {
        "x_min":  np.inf, "x_max": -np.inf,
        "y_min":  np.inf, "y_max": -np.inf,
        "z_min":  np.inf, "z_max": -np.inf,
        "sum_x":  0.0,    "sum_y":  0.0,    "sum_z":  0.0,
        "count":  0,
        "coords": []
    })
    shape_z, shape_y, shape_x = data.shape

    for z in range(shape_z):
        for y in range(shape_y):
            for x in range(shape_x):
                label = data[z,y,x]
                if label == 0:
                    continue
                info = label_info[label]
                # update bounding box
                if x < info["x_min"]: info["x_min"] = x
                if x > info["x_max"]: info["x_max"] = x
                if y < info["y_min"]: info["y_min"] = y
                if y > info["y_max"]: info["y_max"] = y
                if z < info["z_min"]: info["z_min"] = z
                if z > info["z_max"]: info["z_max"] = z

                # sums
                info["sum_x"] += x
                info["sum_y"] += y
                info["sum_z"] += z
                info["count"] += 1

                # store coords (for approximate surface or adjacency)
                info["coords"].append((z,y,x))

    # finalize bounding box, centroid, volume
    for lbl, info in label_info.items():
        c = info["count"]
        info["volume"] = c
        info["centroid"] = (info["sum_z"]/c, info["sum_y"]/c, info["sum_x"]/c)
        info["bbox"] = (
            info["z_min"], info["y_min"], info["x_min"],
            info["z_max"], info["y_max"], info["x_max"]
        )
    return label_info


def approximate_surface_area(data, label, coords):
    face_count = 0
    shape_z, shape_y, shape_x = data.shape

    for (z, y, x) in coords:
        # check 6 neighbors
        for (dz, dy, dx) in [(1,0,0), (-1,0,0),
                             (0,1,0), (0,-1,0),
                             (0,0,1), (0,0,-1)]:
            nz, ny, nx = z+dz, y+dy, x+dx
            if (nz < 0 or nz >= shape_z or
                ny < 0 or ny >= shape_y or
                nx < 0 or nx >= shape_x):
                face_count += 1
            else:
                if data[nz, ny, nx] != label:
                    face_count += 1
    return face_count


def build_adjacency(data):
    adjacency = {}
    shape_z, shape_y, shape_x = data.shape
    for z in range(shape_z):
        for y in range(shape_y):
            for x in range(shape_x):
                L1 = data[z, y, x]
                if L1 == 0:
                    continue
                # only check + directions
                for (dz, dy, dx) in [(1,0,0), (0,1,0), (0,0,1)]:
                    nz, ny, nx = z+dz, y+dy, x+dx
                    if (0 <= nz < shape_z and
                        0 <= ny < shape_y and
                        0 <= nx < shape_x):
                        L2 = data[nz, ny, nx]
                        if L2 != 0 and L2 != L1:
                            adjacency.setdefault(L1, set()).add(int(L2))
                            adjacency.setdefault(L2, set()).add(int(L1))
    return adjacency

def build_adjacency_count(data):
    adjacency = {}
    shape_z, shape_y, shape_x = data.shape

    # Generate shifted versions to check neighbors
    neighbors = [(1,0,0), (0,1,0), (0,0,1)]  # +z, +y, +x directions
    
    for dz, dy, dx in neighbors:
        # Shift the array to get neighbor values
        shifted = np.roll(data, shift=(-dz, -dy, -dx), axis=(0, 1, 2))
        
        # Mask invalid shifts (avoid wrapping around at edges)
        if dz > 0:
            shifted[-dz:, :, :] = 0
        if dy > 0:
            shifted[:, -dy:, :] = 0
        if dx > 0:
            shifted[:, :, -dx:] = 0

        # Get pairs where labels differ and are nonzero
        mask = (data != 0) & (shifted != 0) & (data != shifted)
        L1, L2 = data[mask], shifted[mask]

        # Populate adjacency dictionary
        for l1, l2 in zip(L1, L2):
            adjacency.setdefault(int(l1), {}).setdefault(int(l2), 0)
            adjacency.setdefault(int(l2), {}).setdefault(int(l1), 0)
            adjacency[int(l1)][int(l2)] += 1
            adjacency[int(l2)][int(l1)] += 1

    return adjacency

def extract(path):
    with tiff.TiffFile(path) as tif:
        pages = tif.pages
        z_planes = [page.asarray() for page in pages]
    return np.stack(z_planes, axis=0)

def main(args):
    in_path = args.in_path
    out_path = args.out_path
    out_json_path = args.out_json_path
    print("Starting analysis...")

    results = {}
    previous_centroids = {}
    filelist = sorted(os.listdir(in_path))

    for filename in filelist:  # or all files
        if filename.endswith(".tif"):
            tf_num = int(filename[4:7])
            print("Processing file:", filename)
            file_path = os.path.join(in_path, filename)
            data = tiff.imread(file_path)

            # 1. Gather label info in a single pass
            label_info = gather_label_info(data)
            
            # 3. For each label, fill in metrics
            unique_labels = sorted(label_info.keys())  # sorted for consistency
            current_centroids = {}
            for label_value in unique_labels:
                label_value = int(label_value)
                info = label_info[label_value]
                mask_volume = info["volume"]
                centroid = info["centroid"]
                bounding_box = info["bbox"]
                # bounding box dims
                max_length_z = bounding_box[3] - bounding_box[0] + 1
                max_length_y = bounding_box[4] - bounding_box[1] + 1
                max_length_x = bounding_box[5] - bounding_box[2] + 1

                # approximate surface area from voxel faces
                surf_area = approximate_surface_area(data, label_value, info["coords"])

                # movement from previous frame
                movement = None
                if label_value in previous_centroids:
                    prev_c = previous_centroids[label_value]
                    movement = float(np.linalg.norm(np.array(centroid) - np.array(prev_c)))


                # Store results
                results.setdefault(label_value, {})
                results[label_value][tf_num] = {
                    "surface_area": int(surf_area),
                    "max_length_x": int(max_length_x),
                    "max_length_y": int(max_length_y),
                    "max_length_z": int(max_length_z),
                    "volume": int(mask_volume),
                    "centroid": centroid,
                    "movement": movement,
                }
                current_centroids[label_value] = centroid

            previous_centroids = current_centroids

            out_file_path = os.path.join(out_path, f"t{tf_num:03d}")
            z = zarr.open(out_file_path, mode='w', shape=data.shape, chunks=(100, 100, 100), dtype=data.dtype)
            z[:] = data

    with open(os.path.join(out_json_path, "cell_information.json"), "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run tiff to zarr script for training.")
    parser.add_argument("--in_path", type=str, required=True)    
    parser.add_argument("--out_path", type=str, required=True)    
    parser.add_argument("--out_json_path", type=str, required=True)    
    args = parser.parse_args()
    
    main(args)