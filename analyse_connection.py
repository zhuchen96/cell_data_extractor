import json

# Load JSON file as a dictionary
with open("/work/scratch/zhuchen/SAM-Med3D-Updated/analyse/mask_analysis_results_mem1.json", "r") as file:
    data = json.load(file)

connnections = {}
# Iterate through the first layer
for cell_id, time_frame_info in data.items():
    connnections[str(cell_id)] = {}
    for time_frame, data in time_frame_info.items():
        neighbors = data["adjacency"]
        if neighbors:
            for neibgbor, size in neighbors.items():
                if int(neibgbor) not in connnections[str(cell_id)].keys():
                    connnections[str(cell_id)][int(neibgbor)] = {}
                    connnections[str(cell_id)][int(neibgbor)]['time'] = []
                    connnections[str(cell_id)][int(neibgbor)]['size'] = []
                connnections[str(cell_id)][int(neibgbor)]['time'].append(int(time_frame))
                connnections[str(cell_id)][int(neibgbor)]['size'].append(int(size))
                
# Example usage

with open("/work/scratch/zhuchen/SAM-Med3D-Updated/analyse/mask_analysis_results_mem1_connection.json", "w") as f:
    json.dump(connnections, f, indent=4)
