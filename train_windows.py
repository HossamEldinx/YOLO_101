from ultralytics import YOLO

model = YOLO("yolov8m.yaml")  # build a new model from scratch
# train the model
results = model.train(data="config.yaml", epochs=100, batch=3)
