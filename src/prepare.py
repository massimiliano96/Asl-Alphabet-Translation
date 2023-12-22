import os
import random
import shutil

import yaml

from utils import cv_utils

# Create all paths
list_path = [
    "datasets",
    "datasets/train",
    "datasets/train/images",
    "datasets/train/labels",
    "datasets/validation",
    "datasets/validation/images",
    "datasets/validation/labels",
    "datasets/test",
    "datasets/test/images",
    "datasets/test/labels",
]

# Cleanup dataset directories
for path in list_path:
    if os.path.exists(path):
        os.removedirs(path)
    os.mkdir(path)

# Load parameters for dataset splitting
params = yaml.safe_load(open("params.yaml"))["prepare"]

path_dataset_images = params["image_directory"]
path_dataset_labels = str(params["annotations_directory"] + "/labels")
train_proportion = params["train_proportion"]
validation_proportion = params["validation_proportion"]
test_proportion = params["test_proportion"]
seed = params["seed"]

# Set the seed for the division of dataset
random.seed(seed)

path_train_images = "datasets/train/images"
path_train_labels = "datasets/train/labels"
path_validation_images = "datasets/validation/images"
path_validation_labels = "datasets/validation/labels"
path_test_images = "datasets/test/images"
path_test_labels = "datasets/test/labels"

list_images = os.listdir(path_dataset_images)

# VALIDATION
n_images_validation = round(
    validation_proportion * len(os.listdir(path_dataset_images))
)
list_img_validation = random.sample(list_images, n_images_validation)

for image in list_img_validation:
    name_file_label = image[:-4] + ".txt"
    shutil.copy(f"{path_dataset_images}/{image}", f"{path_validation_images}/{image}")
    if "nothing" not in image:
        shutil.copy(
            f"{path_dataset_labels}/{name_file_label}",
            f"{path_validation_labels}/{name_file_label}",
        )

# TEST
list_images_remained = list(set(list_images) - set(list_img_validation))
n_images_test = round(test_proportion * len(os.listdir(path_dataset_images)))
list_img_test = random.sample(list_images_remained, n_images_test)

for image in list_img_test:
    name_file_label = image[:-4] + ".txt"
    shutil.copy(f"{path_dataset_images}/{image}", f"{path_test_images}/{image}")
    if "nothing" not in image:
        shutil.copy(
            f"{path_dataset_labels}/{name_file_label}",
            f"{path_test_labels}/{name_file_label}",
        )

# TRAINING
list_images_remained1 = list(set(list_images_remained) - set(list_img_test))

for image in list_images_remained1:
    name_file_label = image[:-4] + ".txt"
    shutil.copy(f"{path_dataset_images}/{image}", f"{path_train_images}/{image}")
    if "nothing" not in image:
        shutil.copy(
            f"{path_dataset_labels}/{name_file_label}",
            f"{path_train_labels}/{name_file_label}",
        )

# Load parameters for brightness adjustment
adjust_images_brightness_strategy = params["adjust_images_brightness_strategy"]
gamma = params["gamma"]
percentile = params["percentile"]

# Define Image Brightness Handler
image_brightness_handler = cv_utils.ImageBrightnessHandler(cv_utils.NoneImageStrategy())

# Set the correct brightness adjust strategy
# match is not supported in Python 3.8, use if-else
if adjust_images_brightness_strategy == "All":
    image_brightness_handler.strategy = cv_utils.AllImageStrategy()
elif adjust_images_brightness_strategy == "Dark":
    image_brightness_handler.strategy = cv_utils.DarkImagesStrategy(percentile)

# Perform Brightness adjustment
for dataset in ["train", "validation", "test"]:
    path_dataset_images = str("datasets/" + dataset + "/images")
    image_brightness_handler.adjust_images_brightness(path_dataset_images, gamma)

# Prepare the Yolo config file
# Read classes from classes.txt
with open(str(params["annotations_directory"] + "/classes.txt"), "r") as classes_file:
    classes = [line.strip() for line in classes_file]

# Create YAML data
yaml_data = {
    "train": "train",  # Adjust the paths accordingly
    "val": "validation",
    "names": {index: classname for index, classname in enumerate(classes)},
}

# Write YAML data to yolo-config.yaml
with open("yolo-config.yaml", "w") as yaml_file:
    yaml.dump(yaml_data, yaml_file, default_flow_style=False)

print("yolo-config.yaml has been updated with class information.")
