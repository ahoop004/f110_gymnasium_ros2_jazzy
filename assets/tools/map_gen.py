# generate_straight_corridor.py
from PIL import Image
import numpy as np

width, height = 50, 2000      # 
free = 254                    # free space (white)
occ  =   0                    # occupied (black)

arr = np.full((height, width), free, dtype=np.uint8)
arr[:, :3]   = occ          # left wall
arr[:, -3:]  = occ          # right wall

Image.fromarray(arr).save('/home/aaron/f110_gymnasium_ros2_jazzy/assets/maps/straight_corridor.png')
