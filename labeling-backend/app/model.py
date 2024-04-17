import logging
import os
from typing import List

import cv2
import mlflow
import numpy as np
import requests
import torch
from label_studio_tools.core.label_config import parse_config
from PIL import Image
from requests.auth import HTTPBasicAuth

from .datamodel import Setup, Task
from .utils import download_url, uri_to_url


class Model:
    def __init__(self):
        """Good place to load your model and setup variables"""

        self.project = None
        self.schema = None
        self.hostname = None
        self.access_token = None

        self.user = os.getenv("DAGSHUB_USER_NAME")
        self.token = os.getenv("DAGSHUB_REPO_TOKEN")
        self.repo = os.getenv("DAGSHUB_REPO_NAME")
        self.owner = os.getenv("DAGSHUB_REPO_OWNER")

        # HERE: Load model
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

        # List available models and versions
        model_name = "asl-translation-yolov8"
        models = mlflow.search_model_versions(
            filter_string=f"name='{model_name}'", max_results=10
        )
        
        model_version = "0"
        run_id = ""
        for model in models:
            if model.current_stage == "Staging":
                model_version = model.version
                run_id = model.run_id

        logging.info("Model Version = " + model_version)
        model_uri = f"models:/{model_name}/{model_version}"
        
        
        run = mlflow.get_run(run_id)

        # Retrieve parameters of the run
        params = run.data.params
        self.adjust_images_brightness_strategy = params["adjust_images_brightness_strategy"]
        print(self.adjust_images_brightness_strategy)
            
        mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path="./model")

        self.model = torch.load("./model/data/model.pth")
        self.model_version = model_version

    def setup(self, setup: Setup):
        """Store the setup information sent by Label Studio to the ML backend"""

        self.project = setup.project
        self.parsed_label_config = parse_config(setup.label_schema)
        self.hostname = setup.hostname
        self.access_token = setup.access_token

        from_name, schema = list(self.parsed_label_config.items())[0]
        self.from_name = from_name
        self.to_name = schema["to_name"][0]
        self.labels = schema["labels"]

    def send_predictions(self, result):
        """Send prediction results to Label Studio"""

        url = f"https://dagshub.com/{self.owner}/{self.repo}/annotations/git/api/predictions/"
        auth = HTTPBasicAuth(self.user, self.token)
        res = requests.post(url, auth=auth, json=result)
        logging.warning(result)
        if res.status_code != 200:
            logging.warning(f"{res.status_code}: {res.reason}")

    def predict(self, tasks: List[Task]):
        for task in tasks:
            uri = task.data["image"]
            url = uri_to_url(uri, self.owner, self.repo)
            image_path = download_url(url, self.user, self.token)

            pil_image = Image.open(image_path)
            np_array = np.array(pil_image)
            cv_img = cv2.cvtColor(np_array, cv2.COLOR_RGB2BGR)
            img_h, img_w, _ = cv_img.shape
            results = self.model.predict(cv_img, imgsz=640, conf=0.5)
            img_results = []
            for result in results:
                for i in range(0, len(result.boxes.xywh)):
                    class_id = int(result.boxes.cls[i].item())
                    prob = float(result.boxes.conf[i].item())
                    x = result.boxes.xywh[i][0].item()
                    y = result.boxes.xywh[i][1].item()
                    w = result.boxes.xywh[i][2].item()
                    h = result.boxes.xywh[i][3].item()

                    x = 100 * float(x - w / 2) / img_w
                    y = 100 * float(y - h / 2) / img_h
                    w = 100 * float(w) / img_w
                    h = 100 * float(h) / img_h
                    img_results.append(
                        {
                            "from_name": self.from_name,
                            "to_name": self.to_name,
                            "type": "rectanglelabels",
                            "value": {
                                "rectanglelabels": [self.model.names[class_id]],
                                "x": x,
                                "y": y,
                                "width": w,
                                "height": h,
                            },
                            "score": prob,
                        }
                    )
            result_to_send = {
                "result": img_results,
                "model_version": self.model_version,
                "task": task.id,
            }

            self.send_predictions(result_to_send)
