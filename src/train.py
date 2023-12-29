import os
import shutil

import yaml
from dagshub import dagshub_logger
from ultralytics import YOLO

# Load parameters
params = yaml.safe_load(open("params.yaml"))["train"]

image_size = params["image_size"]
epochs = params["epochs"]
batch_size = params["batch_size"]
init_lr = float(params["init_lr"])
patience = params["patience"]
seed = params["seed"]
directory_output = params["directory_output"]
optimizer = params["optimizer"]

params = yaml.safe_load(open("params.yaml"))["prepare"]

path_dataset_images = params["image_directory"]
path_dataset_labels = str(params["annotations_directory"] + "/labels")
train_proportion = params["train_proportion"]
validation_proportion = params["validation_proportion"]
test_proportion = params["test_proportion"]
seed = params["seed"]

if os.path.exists(directory_output):
    shutil.rmtree(directory_output)
os.mkdir(directory_output)


# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch

# Use the model
results = model.train(
    data="yolo-config.yaml",
    imgsz=image_size,
    epochs=epochs,
    batch=batch_size,
    lr0=init_lr,
    patience=patience,
    seed=seed,
    project=directory_output,
    optimizer=optimizer,
)  # train the model

metrics = {
    key.replace("metrics", "train"): value
    for key, value in results.results_dict.items()
}

with dagshub_logger(
    metrics_path="logs/train_metrics.csv", should_log_hparams=False
) as logger:
    # Metric logging
    logger.log_metrics(metrics)
