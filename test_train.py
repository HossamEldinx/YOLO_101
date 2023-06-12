from ultralytics import YOLO

model = YOLO("best.pt")

model.predict("D:\coding\Yolo\yolo101\imgvald\img2.png",save=True,show=True, save_txt=True, conf=0.4)