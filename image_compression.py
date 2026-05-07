import os
import sys
import numpy as np
import tkinter as tk
from tkinter import filedialog as fd
from PIL import Image
from matplotlib.image import imsave
from matplotlib.colors import LinearSegmentedColormap

"""
When running the code, it opens a window to select the image file
Then, it'll open a window for saving the output image, which should be in the "Output images" folder, because of the sub-folders needed for all the saving
The saved file should be in the parent "Output images" folder, not in the sub-folders with other names
"""

DEBUG = False
root = tk.Tk()
root.withdraw()

# 1. Importing the image file
red_cmap = LinearSegmentedColormap.from_list("red_cmap", ["black", "red"])
green_cmap = LinearSegmentedColormap.from_list("green_cmap", ["black", "green"])
blue_cmap = LinearSegmentedColormap.from_list("blue_cmap", ["black", "blue"])

if not DEBUG:
    available_image_formats = "*.png *.jpg *.jpeg *.webp"

    image_import_file_path = fd.askopenfilename(
        title="Select image file",
        initialdir=os.getcwd(),
        filetypes=[("Image files", available_image_formats)],
    )

    if not image_import_file_path:
        print("No image file selected!", file=sys.stderr)
        sys.exit(-1)

    print(f"Reading from '{image_import_file_path}'...")

    image_export_file_path = fd.asksaveasfilename(
        title="Export image",
        defaultextension=".png",
        initialdir=os.getcwd(),
        filetypes=[("Image files", available_image_formats)],
    )

    print(f"Exporting to '{image_export_file_path}'...")
else:
    # image_import_file_path = "D:/Science Homeschool/Astra Nova/Masa - Class Files - 2025-2026/Spring Term Class Files/Algebra II/FFT & DFT/Image compression/Input images/GyaiyxBWsAA0G7n.jpg"
    # image_export_file_path = "D:/Science Homeschool/Astra Nova/Masa - Class Files - 2025-2026/Spring Term Class Files/Algebra II/FFT & DFT/Image compression/Output images/test.png"
    image_import_file_path = "D:/Science Homeschool/Astra Nova/Masa - Class Files - 2025-2026/Spring Term Class Files/Algebra II/FFT & DFT/Image compression/Input images/IMG_4068.png"
    image_export_file_path = "D:/Science Homeschool/Astra Nova/Masa - Class Files - 2025-2026/Spring Term Class Files/Algebra II/FFT & DFT/Image compression/Output images/test.png"

root.quit()

full_filename = os.path.basename(image_export_file_path)
only_filename, file_type = os.path.splitext(full_filename)


# 2. Reading the image file
image_matrix = np.array(Image.open(image_import_file_path).convert("RGB"))

image_matrix_R = image_matrix[:, :, 0]
image_matrix_G = image_matrix[:, :, 1]
image_matrix_B = image_matrix[:, :, 2]


# 3. Splitting the image and exporting the RGB channels
imsave(os.path.join(os.path.dirname(image_export_file_path), "Uncompressed images", only_filename + "_R_full" + file_type), image_matrix_R, cmap=red_cmap)
imsave(os.path.join(os.path.dirname(image_export_file_path), "Uncompressed images", only_filename + "_G_full" + file_type), image_matrix_G, cmap=green_cmap)
imsave(os.path.join(os.path.dirname(image_export_file_path), "Uncompressed images", only_filename + "_B_full" + file_type), image_matrix_B, cmap=blue_cmap)

recombined_image_matrix = np.stack((image_matrix_R, image_matrix_G, image_matrix_B), axis=2)

Image.fromarray(np.clip(recombined_image_matrix, 0, 255).astype(np.uint8)).save(os.path.join(os.path.dirname(image_export_file_path), "Uncompressed images", only_filename + "_RGB_full" + file_type))


# 4. Getting compression amount
if not DEBUG:
    percentage_str = input("Input % of data to keep: ")

    data_percentage = 50

    try:
        data_percentage = float(percentage_str)
    except:
        print(f"Inputted number was not a float, data percentage to keep was set to {data_percentage}")
else:
    data_percentage = 10

data_percentage /= 100


# 4a. Raw compression (wrong way)
image_matrix_R_sorted = np.sort(image_matrix_R.reshape(-1))
image_matrix_G_sorted = np.sort(image_matrix_G.reshape(-1))
image_matrix_B_sorted = np.sort(image_matrix_B.reshape(-1))

thresh_R = image_matrix_R_sorted[int(np.floor((1 - data_percentage) * len(image_matrix_R_sorted)))]
thresh_G = image_matrix_G_sorted[int(np.floor((1 - data_percentage) * len(image_matrix_G_sorted)))]
thresh_B = image_matrix_B_sorted[int(np.floor((1 - data_percentage) * len(image_matrix_B_sorted)))]

image_matrix_R_filtered_real = image_matrix_R * (np.abs(image_matrix_R) > thresh_R)
image_matrix_G_filtered_real = image_matrix_G * (np.abs(image_matrix_G) > thresh_G)
image_matrix_B_filtered_real = image_matrix_B * (np.abs(image_matrix_B) > thresh_B)

imsave(os.path.join(os.path.dirname(image_export_file_path), "Compressed raw images", only_filename + "_R_compressed_raw" + file_type), image_matrix_R_filtered_real, cmap=red_cmap)
imsave(os.path.join(os.path.dirname(image_export_file_path), "Compressed raw images", only_filename + "_G_compressed_raw" + file_type), image_matrix_G_filtered_real, cmap=green_cmap)
imsave(os.path.join(os.path.dirname(image_export_file_path), "Compressed raw images", only_filename + "_B_compressed_raw" + file_type), image_matrix_B_filtered_real, cmap=blue_cmap)

recombined_image_matrix = np.stack((image_matrix_R_filtered_real, image_matrix_G_filtered_real, image_matrix_B_filtered_real), axis=2)

Image.fromarray(np.clip(recombined_image_matrix, 0, 255).astype(np.uint8)).save(os.path.join(os.path.dirname(image_export_file_path), "Compressed raw images", only_filename + "_RGB_compressed_raw" + file_type))


# 4b. FFT compression (right way)
image_matrix_R_FFT2 = np.fft.fft2(image_matrix_R)
image_matrix_G_FFT2 = np.fft.fft2(image_matrix_G)
image_matrix_B_FFT2 = np.fft.fft2(image_matrix_B)

viewable_image_matrix_R_FFT2 = np.log(1 + np.abs(np.fft.fftshift(image_matrix_R_FFT2)))
viewable_image_matrix_G_FFT2 = np.log(1 + np.abs(np.fft.fftshift(image_matrix_G_FFT2)))
viewable_image_matrix_B_FFT2 = np.log(1 + np.abs(np.fft.fftshift(image_matrix_B_FFT2)))

normalized_image_matrix_R_FFT2 = ((viewable_image_matrix_R_FFT2 - viewable_image_matrix_R_FFT2.min()) / (viewable_image_matrix_R_FFT2.max() - viewable_image_matrix_R_FFT2.min()) * 255).astype(np.uint8)
normalized_image_matrix_G_FFT2 = ((viewable_image_matrix_G_FFT2 - viewable_image_matrix_G_FFT2.min()) / (viewable_image_matrix_G_FFT2.max() - viewable_image_matrix_G_FFT2.min()) * 255).astype(np.uint8)
normalized_image_matrix_B_FFT2 = ((viewable_image_matrix_B_FFT2 - viewable_image_matrix_B_FFT2.min()) / (viewable_image_matrix_B_FFT2.max() - viewable_image_matrix_B_FFT2.min()) * 255).astype(np.uint8)

imsave(os.path.join(os.path.dirname(image_export_file_path), "FFT2 full", only_filename + "_R_full_FFT2" + file_type), normalized_image_matrix_R_FFT2, cmap=red_cmap)
imsave(os.path.join(os.path.dirname(image_export_file_path), "FFT2 full", only_filename + "_G_full_FFT2" + file_type), normalized_image_matrix_G_FFT2, cmap=green_cmap)
imsave(os.path.join(os.path.dirname(image_export_file_path), "FFT2 full", only_filename + "_B_full_FFT2" + file_type), normalized_image_matrix_B_FFT2, cmap=blue_cmap)

recombined_image_matrix = np.stack((normalized_image_matrix_R_FFT2, normalized_image_matrix_G_FFT2, normalized_image_matrix_B_FFT2), axis=2)

Image.fromarray(np.clip(recombined_image_matrix, 0, 255).astype(np.uint8)).save(os.path.join(os.path.dirname(image_export_file_path), "FFT2 full", only_filename + "_RGB_full_FFT2" + file_type))

image_matrix_R_sorted = np.sort(np.abs(image_matrix_R_FFT2).reshape(-1))
image_matrix_G_sorted = np.sort(np.abs(image_matrix_G_FFT2).reshape(-1))
image_matrix_B_sorted = np.sort(np.abs(image_matrix_B_FFT2).reshape(-1))

thresh_R = image_matrix_R_sorted[int(np.floor((1 - data_percentage) * len(image_matrix_R_sorted)))]
thresh_G = image_matrix_G_sorted[int(np.floor((1 - data_percentage) * len(image_matrix_G_sorted)))]
thresh_B = image_matrix_B_sorted[int(np.floor((1 - data_percentage) * len(image_matrix_B_sorted)))]

image_matrix_R_filtered = image_matrix_R_FFT2 * (np.abs(image_matrix_R_FFT2) > thresh_R)
image_matrix_G_filtered = image_matrix_G_FFT2 * (np.abs(image_matrix_G_FFT2) > thresh_G)
image_matrix_B_filtered = image_matrix_B_FFT2 * (np.abs(image_matrix_B_FFT2) > thresh_B)

viewable_image_matrix_R_FFT2 = np.log(1 + np.abs(np.fft.fftshift(image_matrix_R_filtered)))
viewable_image_matrix_G_FFT2 = np.log(1 + np.abs(np.fft.fftshift(image_matrix_G_filtered)))
viewable_image_matrix_B_FFT2 = np.log(1 + np.abs(np.fft.fftshift(image_matrix_B_filtered)))

normalized_image_matrix_R_FFT2 = ((viewable_image_matrix_R_FFT2 - viewable_image_matrix_R_FFT2.min()) / (viewable_image_matrix_R_FFT2.max() - viewable_image_matrix_R_FFT2.min()) * 255).astype(np.uint8)
normalized_image_matrix_G_FFT2 = ((viewable_image_matrix_G_FFT2 - viewable_image_matrix_G_FFT2.min()) / (viewable_image_matrix_G_FFT2.max() - viewable_image_matrix_G_FFT2.min()) * 255).astype(np.uint8)
normalized_image_matrix_B_FFT2 = ((viewable_image_matrix_B_FFT2 - viewable_image_matrix_B_FFT2.min()) / (viewable_image_matrix_B_FFT2.max() - viewable_image_matrix_B_FFT2.min()) * 255).astype(np.uint8)

imsave(os.path.join(os.path.dirname(image_export_file_path), "FFT2 filtered", only_filename + "_R_filtered_FFT2" + file_type), normalized_image_matrix_R_FFT2, cmap=red_cmap)
imsave(os.path.join(os.path.dirname(image_export_file_path), "FFT2 filtered", only_filename + "_G_filtered_FFT2" + file_type), normalized_image_matrix_G_FFT2, cmap=green_cmap)
imsave(os.path.join(os.path.dirname(image_export_file_path), "FFT2 filtered", only_filename + "_B_filtered_FFT2" + file_type), normalized_image_matrix_B_FFT2, cmap=blue_cmap)

recombined_image_matrix = np.stack((normalized_image_matrix_R_FFT2, normalized_image_matrix_G_FFT2, normalized_image_matrix_B_FFT2), axis=2)

Image.fromarray(np.clip(recombined_image_matrix, 0, 255).astype(np.uint8)).save(os.path.join(os.path.dirname(image_export_file_path), "FFT2 filtered", only_filename + "_RGB_filtered_FFT2" + file_type))

image_matrix_R_filtered_real = np.fft.ifft2(image_matrix_R_filtered).real
image_matrix_G_filtered_real = np.fft.ifft2(image_matrix_G_filtered).real
image_matrix_B_filtered_real = np.fft.ifft2(image_matrix_B_filtered).real

imsave(os.path.join(os.path.dirname(image_export_file_path), "Compressed FFT2 images", only_filename + "_R_compressed_FFT2" + file_type), image_matrix_R_filtered_real, cmap=red_cmap)
imsave(os.path.join(os.path.dirname(image_export_file_path), "Compressed FFT2 images", only_filename + "_G_compressed_FFT2" + file_type), image_matrix_G_filtered_real, cmap=green_cmap)
imsave(os.path.join(os.path.dirname(image_export_file_path), "Compressed FFT2 images", only_filename + "_B_compressed_FFT2" + file_type), image_matrix_B_filtered_real, cmap=blue_cmap)

recombined_image_matrix = np.stack((image_matrix_R_filtered_real, image_matrix_G_filtered_real, image_matrix_B_filtered_real), axis=2)

Image.fromarray(np.clip(recombined_image_matrix, 0, 255).astype(np.uint8)).save(os.path.join(os.path.dirname(image_export_file_path), "Compressed FFT2 images", only_filename + "_RGB_compressed_FFT2" + file_type))