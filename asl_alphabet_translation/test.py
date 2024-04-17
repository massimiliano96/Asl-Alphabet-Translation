import mlflow
import mlflow.pytorch
import yaml
from ultralytics import YOLO

from utils import plot_utils

# Read the run ID from the file
with open("run_id.txt", "r") as file:
    run_id = file.read().strip()
mlflow.start_run(run_id=run_id)

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

mlflow.log_metric("test-precision", results.results_dict["metrics/precision(B)"])
mlflow.log_metric("test-recall", results.results_dict["metrics/recall(B)"])
mlflow.log_metric("test-mAP50", results.results_dict["metrics/mAP50(B)"])
mlflow.log_metric("test-mAP50-95", results.results_dict["metrics/mAP50-95(B)"])
mlflow.log_metric("test-fitness", results.results_dict["fitness"])

model_to_log = YOLO("output/train/weights/best.pt")
mlflow.pytorch.log_model(
    pytorch_model=model_to_log,
    artifact_path="output/train/weights/best.pt",
    registered_model_name="asl-translation-yolov8",
)

mlflow.end_run()
