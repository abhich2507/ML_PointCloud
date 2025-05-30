from numba import jit
import ROOT
import pandas as pd
from collections import defaultdict
import pickle
import time
from tqdm import tqdm
import os

# User input start and end index


@jit(nopython=True)
def convert_pos(cub_idx, cell_idx, size=(1, 1,1)):
    """
    Converts cublet and cell indices into 3D grid positions with scaling.
    """
    layer_cublet_idx = cub_idx % 100
    x_cub = layer_cublet_idx % 10
    y_cub = 9 - layer_cublet_idx // 10
    z_cub = cub_idx // 100
    
    layer_cell_idx = cell_idx % 100
    x_cell_temp = layer_cell_idx % 10
    y_cell_temp = 9 - layer_cell_idx // 10
    z_cell_temp = cell_idx // 100
    
    x_cell = x_cub * 10 + x_cell_temp
    y_cell = y_cub * 10 + y_cell_temp
    z_cell = z_cub * 10 + z_cell_temp

    # Scaling the grid
    new_x = int(x_cell // (100 // size[0]))
    new_y = int(y_cell // (100 // size[1]))
    new_z = int(z_cell // (200 // size[2])) ## 100 / 100 = 1 , it is required to scale the z axis to 200.

    return [new_x, new_y, new_z]



import pandas as pd
from collections import defaultdict

def process_event(event):
    
    cell_energy_dict = defaultdict(float)  # To store total energy per cell
    cell_time_dict = defaultdict(float)    # To store sum of times per cell
    cell_interactions_count = defaultdict(int)  # To count number of interactions per cell

    for j in range(event.Tinteractions_in_event):
        # Get cell coordinates (x, y, z)
        x, y, z = convert_pos(event.Tcublet_idx[j], event.Tcell_idx[j])
        
        # Get energy and time
        energy = event.Tedep[j]
        time = event.Tglob_t[j]  # Assuming this is the global time for the interaction
        
        # Accumulate total energy for the cell
        cell_energy_dict[(x, y, z)] += energy
        
        # Accumulate sum of time for the cell
        cell_time_dict[(x, y, z)] += time*energy
        
        # Count the number of interactions for the cell
        cell_interactions_count[(x, y, z)] += 1

    # Create a DataFrame for the current event
    
    df_event = pd.DataFrame([{
        "x": key[0],
        "y": key[1],
        "z": key[2],
        "total_energy": cell_energy_dict[key],
        "mean_time": cell_time_dict[key] / (cell_energy_dict[key]) # Energy weighted mean time
    } for key in cell_energy_dict.keys() if cell_energy_dict[key] > 0])

    return df_event

df_events = []

def process_events_in_range(index):
    t = int(index)
   # file = ROOT.TFile.Open(f"/mnt/c/Users/hnayak/Documents/Pion_100GeV_Col/result_pion_{t}.root")
    file = ROOT.TFile.Open(f"/mnt/c/Users/hnayak/OneDrive - University of Tennessee/Pion_25GeV_Col/result_pion_{t}.root")
    # file=ROOT.TFile.Open(f"/mnt/c/Users/hnayak/OneDrive - University of Tennessee/Pion_100GeV_Col/result_pion_{t}.root")
    tree = file.Get("outputTree")
    total_entries = tree.GetEntries()
    print(f"Total Entries: {total_entries}")


    for i, event in enumerate(tqdm(tree, total=total_entries, desc="Processing Events", unit="event", ascii=True)):
        df_event = process_event(event)
        df_events.append(df_event)

    with open(f"/mnt/c/Users/hnayak/Documents/25GeV/PKL_pion_25GeV_1/pion{t}.pkl", "wb") as f:
        pickle.dump(df_events, f)

    print(f"proton{t}.pkl created")
    # root_file_path = f"/mnt/c/Users/hnayak/Documents/Pion_100Gev_Col/result_pion_{t}.root"
    # if os.path.exists(root_file_path):
    #     os.remove(root_file_path)
    #     print(f"Deleted ROOT file: result_proton_{t}.root")
    # else:
    #     print(f"ROOT file not found: result_proton_{t}.root")

import sys
index = sys.argv[1]
print(index)

process_events_in_range(index)







