import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

def generate_visual_legend(config_map, patch_size=50):
    """Creates an image showing the color key and class names."""
    # Filter out the 'unlabeled' if you don't want it in the legend
    labels_to_show = [v for k, v in config_map.items() if k != 255]

    # Calculate dimensions
    height = len(labels_to_show) * patch_size
    width = 300
    legend_img = np.ones((height, width, 3), dtype=np.uint8) * 255 # White background

    for i, info in enumerate(labels_to_show):
        y_offset = i * patch_size
        color = info["display_rgb"] # Assumes RGB

        # Draw the color square
        # cv2.rectangle(img, pt1, pt2, color, thickness)
        # Note: we use -1 for thickness to "fill" the square
        cv2.rectangle(legend_img, (5, y_offset + 5), (patch_size - 5, y_offset + patch_size - 5),
                      color[::-1], -1) # Flip color to BGR for OpenCV

        # Write the text
        cv2.putText(legend_img, info["name"], (patch_size + 10, y_offset + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    return legend_img

# Usage:

class_map = {
    0:{
        "mask_rgb": [0,0,0],
        "display_rgb": [21, 171, 234], #Light Blue
        "name": "soil"
    },
    1:{
        "mask_rgb": [1,1,1],
        "display_rgb": [191, 21, 234], #Purple
        "name": "bedrock"
    },
    2:{
        "mask_rgb": [2,2,2],
        "display_rgb": [234, 84, 21], #Light Orange
        "name": "sand"
    },
    3:{
        "mask_rgb": [3,3,3],
        "display_rgb": [64, 234, 21], #Green
        "name": "big rock"
    },
    255:{
        "mask_rgb": [255,255, 255],
        "display_rgb": [255,255, 255], #White
        "name": "unlabeled"
    },
}

legend = generate_visual_legend(class_map)

img = cv2.imread('test_mask.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

visual_img=np.zeros_like(img)

for class_id, info in class_map.items():
    target_color=info["mask_rgb"]
    display_color=info["display_rgb"]

    #axis -1 is the RGB information axis
    match_region = np.all(img == target_color, axis=-1)
    #match_3D = np.stack([match_region]*3, axis=-1)
    visual_img[match_region] = display_color

background = cv2.imread('test_image.jpeg')
visual_bgr = cv2.cvtColor(visual_img, cv2.COLOR_RGB2BGR)

alpha=0.3
overlay = cv2.addWeighted(visual_bgr, alpha, background, 1-alpha, 0)

