import json

# Load JSON from a local file
with open('/work/scratch/zhuchen/cell_data_extractor/json_files_test_nuc/cell_information.json', 'r') as f:   # <-- change 'your_file.json' to your actual filename
    data = json.load(f)

# Collect all "d" values regardless of frame or cell id
d_values = []

for frame_id, cells in data["cell_information"].items():
    for cell_id, cell_info in cells.items():
        d_values.append(cell_info["d"])

# Calculate average
average_d = sum(d_values) / len(d_values) if d_values else 0

# Print result
print("All d values:", d_values)
print("Average d:", average_d)