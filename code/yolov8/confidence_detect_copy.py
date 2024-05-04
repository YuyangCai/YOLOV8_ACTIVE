import yaml
import random
import os
from ultralytics import YOLO
import json
import numpy as np
from pathlib import Path
import shutil

# 设置基本路径和文件名
BASE_PATH = '/home/cyy/code/yolov8/user_yaml' 
ORIGINAL_YAML = 'coco.yaml'
TEMP_YAML = 'temp_coco.yaml'
IMAGE_LIST = 'temp_image_list.txt'
VALIDATION_LIST = 'validation_image_list.txt'
PREDICTIONS_FILE = 'predictions.json'

samples = 10000  #设置每轮训练的样本数
SAMPLES_DIR = Path('/home/cyy/code/samples/test11')

cycles = 9


def load_yaml(filepath):
    with open(filepath, 'r') as file:
        return yaml.safe_load(file)

def save_yaml(data, filepath):
    with open(filepath, 'w') as file:
        yaml.safe_dump(data, file)

def save_image_list(image_paths, filepath):
    with open(filepath, 'w') as file:
        for path in image_paths:
            file.write(f"{path}\n")

def get_image_paths(directory):
    return [os.path.join(directory, img) for img in os.listdir(directory)]

def select_images(image_paths, samples, seed=42):
    random.seed(seed)
    selected = random.sample(image_paths, samples)
    for img_path in selected:
        shutil.copy(img_path, SAMPLES_DIR)

    return selected

def train_model(config_path, epochs, imgsz, batch, lr0, val=True):
    model = YOLO("yolov8n.yaml")
    model.train(data=config_path, epochs=epochs, imgsz=imgsz, batch=batch, lr0=lr0 ,val=val)
    return model

def process_predictions(prediction_file_path, image_paths):
    with open(prediction_file_path, 'r') as file:
        predictions_data = json.load(file)

    no_detections = []
    confidence_data = []

    for item in predictions_data:
        predictions = item['predictions']
        image_index = item['image_index']

        if not predictions:
            no_detections.append(image_index)
        else:
            predictions_array = np.array(predictions)
            average_confidence = np.mean(predictions_array[:, 4])
            confidence_data.append((image_index, average_confidence))

    if len(no_detections) >= samples:
        selected_indices = no_detections[:samples]
    else:
        confidence_data.sort(key=lambda x: x[1])
        samples_needed = samples - len(no_detections)
        selected_indices = no_detections + [x[0] for x in confidence_data[:samples_needed]]

    selected_image_paths = [image_paths[i] for i in selected_indices]
    remaining_images = [img for i, img in enumerate(image_paths) if i not in selected_indices]
    
    for img_path in selected_image_paths:
        shutil.copy(img_path,SAMPLES_DIR)

    return selected_image_paths, remaining_images

# 加载初始配置
config = load_yaml(os.path.join(BASE_PATH, ORIGINAL_YAML))
all_images = get_image_paths(config['train'])
training_images = all_images

# 预先选择图片和加载数据
selected_images = select_images(training_images, samples)
remaining_images = list(set(all_images) - set(selected_images))

fixed_weights_dir = Path('/home/cyy/code/runs/detect/weights')

fixed_weights_dir.mkdir(parents= True, exist_ok=True)

fixed_last_weights_path = fixed_weights_dir / 'last.pt'


for cycle in range(cycles):
    print(f"Cycle {cycle + 1}/cycles")

    if cycle != 0:
        # 更新训练集
        training_images = remaining_images
        selected_images, remaining_images = process_predictions(os.path.join('predictions.json'), list(training_images))
        remaining_images = set(remaining_images)  # Convert list to set for next iteration

    # 更新文件列表
    save_image_list(list(selected_images), os.path.join(BASE_PATH, IMAGE_LIST))
    save_image_list(list(remaining_images), os.path.join(BASE_PATH, VALIDATION_LIST))

    # 更新配置文件
    config['train'] = os.path.join(BASE_PATH, IMAGE_LIST)
    config['val'] = os.path.join(BASE_PATH, VALIDATION_LIST) 
    save_yaml(config, os.path.join(BASE_PATH, TEMP_YAML))

    # 训练模型
    if cycle == 0:
        model = train_model(os.path.join(BASE_PATH, TEMP_YAML), 150, 224, 16, 0.01, val=True)
    else:
        model.train(data=os.path.join(BASE_PATH, TEMP_YAML), model=str(fixed_last_weights_path), epochs=150, imgsz=224, batch=16, lr0=0.01, cache=True, resume=True)

    # Update weights
    latest_train_dir = max(Path('/home/cyy/code/runs/detect').glob('train*'), key=os.path.getmtime)
    latest_weights_path = latest_train_dir / 'weights/last.pt'
    shutil.copy(latest_weights_path, fixed_last_weights_path)

model = YOLO('/home/cyy/code/random_runs/detect/weights/last.pt')
validation_results = model.val(data='/home/cyy/code/yolov8/user_yaml/coco.yaml',
                               
                               imgsz=224,
                               batch=16,
                               conf=0.25,
                               iou=0.6,
                               device=0)
