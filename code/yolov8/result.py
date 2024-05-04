import yaml
import random
import os
from ultralytics import YOLO
import json
import numpy as np
from pathlib import Path
import shutil

model = YOLO('/home/cyy/code/random_runs/detect/weights/last.pt')
metrics = model.val(data='/home/cyy/code/yolov8/user_yaml/coco.yaml',
                            
                            imgsz=224,
                            batch=16,
                            conf=0.25,
                            iou=0.6,
                            device=0)

metrics.box.map