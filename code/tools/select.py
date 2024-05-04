import os
import shutil
from random import sample

# Define the paths for the two directories
dir_with_more_images = '/home/cyy/code/datasets/coco/images/train2017'  # Path to the directory with more images
dir_with_selected_images = '/home/cyy/code/datasets/coco_for_al/images/train'  # Path to the directory with selected images

# Get the list of images in both directories
images_in_more = set(os.listdir(dir_with_more_images))
images_in_selected = set(os.listdir(dir_with_selected_images))

# Calculate the difference set to find images that are not in the selected images directory
images_to_select_from = images_in_more - images_in_selected

# Randomly sample 4630 images from the difference set
images_to_add = sample(images_to_select_from, 4630)

# Copy the selected images to the selected images directory
for image in images_to_add:
    shutil.copy(os.path.join(dir_with_more_images, image),
                os.path.join(dir_with_selected_images, image))