from pyglobal import *
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def calculate_r_channel_areas(image_path, class_dic):
    """
    Calculates the relative area for specific R values in an RGB image.

    :param image_path: Path to the PNG file.
    :param class_dic: A dictonary with RGB values as keys, and the label named saved as ["name"]
    as a second layer key
    :return: Dictionary { R_value: relative_area }
    """

    target_r_values = list(class_dic.keys())

    #Load image and ensure it is in RGB mode
    with Image.open(image_path) as img:
        img_rgb = img.convert('RGB')

    # Convert to NumPy array
    data = np.array(img_rgb)

    # Extract only the Red channel (Index 0)
    r_channel = data[:, :, 0]

    # Total number of pixels
    total_pixels = r_channel.size

    results = {}
    for r_val in target_r_values:
        # Count pixels matching the specific R value
        count = np.sum(r_channel == r_val)
        # Calculate relative area (0.0 to 1.0)
        results[class_dic[r_val]["name"]] = count / total_pixels

    return results

def aggregate_folder_r_areas(folder_path, class_dic, sample_limit=None):
    """
    Processes a random subset of PNGs in a folder and calculates global relative area.

    :param folder_path: Path to the directory.
    :param class_dic: { R_value: {"name": "label"} }
    :param sample_limit: (int) Max number of images to process. If None, processes all.
    """
    # 1. Gather all PNG files
    all_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]

    if not all_files:
        print("No PNG files found.")
        return {}

    # 2. Select the subset
    if sample_limit and sample_limit < len(all_files):
        files_to_process = random.sample(all_files, sample_limit)
        print(f"Processing a random subset of {sample_limit} images...")
    else:
        files_to_process = all_files
        print(f"Processing all {len(all_files)} images...")

    # Initialize counters
    global_counts = {class_dic[r]["name"]: 0 for r in class_dic}
    total_pixels_dataset = 0

    # 3. Process the selected files
    for filename in files_to_process:
        file_path = os.path.join(folder_path, filename)

        try:
            with Image.open(file_path) as img:
                img_rgb = img.convert('RGB')
                r_channel = np.array(img_rgb)[:, :, 0]

                total_pixels_dataset += r_channel.size

                for r_val in class_dic:
                    label = class_dic[r_val]["name"]
                    global_counts[label] += np.sum(r_channel == r_val)
        except Exception as e:
            print(f"Could not process {filename}: {e}")

    # 4. Calculate final percentages
    if total_pixels_dataset == 0:
        return {}

    return {label: count / total_pixels_dataset for label, count in global_counts.items()}



def plot_pizza_graph(data_dict, palette_dict=None, title="Relative Areas", output_filename=""):
    """
    Creates a pie chart with a separate legend box containing names and percentages.
    """
    labels = list(data_dict.keys())
    sizes = list(data_dict.values())
    total = sum(sizes)

    # Handle Colors
    if palette_dict:
        # Normalize RGB (0-255) to (0-1) for Matplotlib
        colors = [tuple(c/255.0 for c in palette_dict[name]) for name in labels]
    else:
        colors = plt.cm.Pastel1.colors

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    # Create the pie chart (without autopct inside the slices)
    wedges, _ = ax.pie(
        sizes,
        colors=colors,
        startangle=140,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
    )

    # Create custom labels for the legend: "Name: 15.0%"
    legend_labels = [f'{l}: {(s/total)*100:1.1f}%' for l, s in zip(labels, sizes)]

    # Add the legend box to the side
    ax.legend(
        wedges,
        legend_labels,
        title="Categories",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1) # This moves the box outside the plot
    )

    ax.set_title(title, pad=20, fontsize=14)
    plt.axis('equal')
    if output_filename:
        plt.savefig(output_filename)
        plt.close()
    else:
        plt.show()




dic = aggregate_folder_r_areas("../AI4Mars_Data/msl_nav/labels_train/", glb.labels_nav)
plot_pizza_graph(dic, glb.pallet_nav, "Relative Label Area in m2020 Nav Dataset", "msl_nav_train.png")
