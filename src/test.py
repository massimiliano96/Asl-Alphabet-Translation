import yaml
from ultralytics import YOLO

from utils import plot_utils

# Load parameters
params = yaml.safe_load(open("params.yaml"))["test"]
directory_output = params["directory_output"]

# Import the model
model = YOLO("output/train/weights/best.pt", task="detect")

# Test the model on the test set (workaround since is still not implemented on yolo)
results = model.val(data="yolo-config-test.yaml", project="output")

values = [
    round(results.results_dict["metrics/mAP50(B)"], 2),
    round(results.results_dict["metrics/mAP50-95(B)"], 2),
    round(results.results_dict["fitness"], 2),
]
categories = ["mAP50(B)", "mAP50-95(B)", "fitness"]
plot_utils.plot_metrics(categories, values)
