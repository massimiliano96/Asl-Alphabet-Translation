from ultralytics import YOLO

model = YOLO("outputs/train/weights/best.pt")

test_results = model.predict("dataset/test/images")
