import os
import shutil

import mlflow
import yaml
from ultralytics import YOLO, settings
from ultralytics.utils import LOGGER, RUNS_DIR, colorstr

PREFIX = colorstr("MLflow: ")


def sanitize_values(x):
    return {k.replace("(", "").replace(")", ""): float(v) for k, v in x.items()}


SANITIZE = sanitize_values


settings.update({"mlflow": False})

"""
These methods are crucial for customizing the behavior of Ultralytics YOLOv8 MLflow integration.
The default on_train_end callback, by default, concludes the MLflow run.
However, to include the test metrics in the same run,
I have made modifications to this callback to address this specific requirement.
"""


def on_pretrain_routine_end(trainer):
    """Log training parameters to MLflow at the end of the pretraining routine."""
    global mlflow

    uri = os.environ.get("MLFLOW_TRACKING_URI") or str(RUNS_DIR / "mlflow")
    LOGGER.debug(f"{PREFIX} tracking uri: {uri}")
    mlflow.set_tracking_uri(uri)

    # Set experiment and run names
    experiment_name = (
        os.environ.get("MLFLOW_EXPERIMENT_NAME")
        or trainer.args.project
        or "/Shared/YOLOv8"
    )
    run_name = os.environ.get("MLFLOW_RUN") or trainer.args.name
    mlflow.set_experiment(experiment_name)

    mlflow.autolog()
    try:
        active_run = mlflow.active_run() or mlflow.start_run(run_name=run_name)
        LOGGER.info(f"{PREFIX}logging run_id({active_run.info.run_id}) to {uri}")
        mlflow.log_params(dict(trainer.args))
    except Exception as e:
        LOGGER.warning(
            f"{PREFIX}WARNING ⚠️ Failed to initialize: {e}\n"
            f"{PREFIX}WARNING ⚠️ Not tracking this run"
        )


def on_train_epoch_end(trainer):
    """Log training metrics at the end of each train epoch to MLflow."""
    if mlflow:
        mlflow.log_metrics(
            metrics=SANITIZE(trainer.label_loss_items(trainer.tloss, prefix="train")),
            step=trainer.epoch,
        )
        mlflow.log_metrics(metrics=SANITIZE(trainer.lr), step=trainer.epoch)


def on_fit_epoch_end(trainer):
    """Log training metrics at the end of each fit epoch to MLflow."""
    if mlflow:
        mlflow.log_metrics(metrics=SANITIZE(trainer.metrics), step=trainer.epoch)


def on_train_end(trainer):
    """Log model artifacts at the end of the training."""
    if mlflow:
        mlflow.log_artifact(
            str(trainer.best.parent)
        )  # log save_dir/weights directory with best.pt and last.pt
        for f in trainer.save_dir.glob("*"):  # log all other files in save_dir
            if f.suffix in {".png", ".jpg", ".csv", ".pt", ".yaml"}:
                mlflow.log_artifact(str(f))

        LOGGER.info(f"{PREFIX}results logged to {mlflow.get_tracking_uri()}\n")


# Read the run ID from the file
with open("run_id.txt", "r") as file:
    run_id = file.read().strip()

mlflow.start_run(run_id=run_id)

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

model.add_callback("on_train_end", on_train_end)
model.add_callback("on_pretrain_routine_end", on_pretrain_routine_end)
model.add_callback("on_train_epoch_end", on_train_epoch_end)
model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

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

# log params with mlflow
mlflow.log_param("image_size", image_size)
mlflow.log_param("epochs", epochs)
mlflow.log_param("batch", batch_size)
mlflow.log_param("lr0", init_lr)
mlflow.log_param("patience", patience)
mlflow.log_param("optimizer", optimizer)
