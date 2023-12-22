import yaml
import os
from ultralytics import YOLO

# Load parameters
params = yaml.safe_load(open("params.yaml"))["train"]

image_size = params['image_size']
epochs = params['epochs']
batch_size = params['batch_size']
init_lr = float(params['init_lr'])
patience = params['patience']
seed = params['seed']
directory_output = params['directory_output']
optimizer = params['optimizer']


# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch

# Use the model
results = model.train(
    data = "yolo-config.yaml",
    imgsz = image_size,
    epochs = epochs,
    batch = batch_size,
    lr0 = init_lr,
    patience = patience,
    seed = seed,
    project = directory_output,
    optimizer = optimizer
)  # train the model

model.export(format='onnx', imgsz=image_size)

# # Remove the chache file that create a problem with the dvc pipeline
# file_paths = ['datasets/train/labels.cache', 'datasets/validation/labels.cache']
# for path in file_paths:
#     if os.path.exists(path):
#         os.remove(path)