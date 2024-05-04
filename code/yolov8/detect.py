from ultralytics import YOLO

model = YOLO('/home/cyy/code/yolov8/ultralytics/cfg/models/v8/yolov8.yaml')


results = model.train(data='/home/cyy/code/yolov8/user_yaml/coco.yaml', epochs=150, imgsz=240, device=0,lr0=0.01)