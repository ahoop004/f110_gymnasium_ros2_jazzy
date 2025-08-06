import pandas as pd
import numpy as np

# Path to your centerline CSV
centerline_csv = "/home/aaron/f110_gymnasium_ros2_jazzy/rl_training/maps/cenerlines/Shanghai_map.csv"
num_waypoints = 100  # adjust as needed

# Load x and y coordinates from the CSV, skipping comment/blank lines
coords = []
with open(centerline_csv, "r") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split(',')
        # the first two columns are x and y
        if len(parts) >= 2:
            try:
                x = float(parts[0])
                y = float(parts[1])
                coords.append([x, y])
            except ValueError:
                pass

coords = np.array(coords)

# Compute cumulative arc-length along the path
seg_lengths = np.linalg.norm(np.diff(coords, axis=0), axis=1)
cum_lengths = np.concatenate(([0], np.cumsum(seg_lengths)))
total_length = cum_lengths[-1]

# Generate target distances for waypoints
target_lengths = np.linspace(0, total_length, num_waypoints, endpoint=False)
waypoints = []
idx = 0
for tlen in target_lengths:
    # find the segment containing tlen
    while idx < len(cum_lengths) - 1 and cum_lengths[idx + 1] < tlen:
        idx += 1
    start = coords[idx]
    end = coords[idx + 1] if idx + 1 < len(coords) else coords[0]  # wrap around
    seg_len = np.linalg.norm(end - start)
    if seg_len == 0:
        waypoint = start
    else:
        # Interpolate along the segment
        ratio = (tlen - cum_lengths[idx]) / seg_len
        waypoint = start + ratio * (end - start)
    waypoints.append(waypoint)

# Save the waypoints to a CSV file
waypoints = np.array(waypoints)
pd.DataFrame(waypoints).to_csv("/home/aaron/f110_gymnasium_ros2_jazzy/rl_training/maps/cenerlines/Shanghai_waypoints.csv", index=False, header=False)
