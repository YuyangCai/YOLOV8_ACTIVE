import yaml
import random
import os
import json
import numpy as np
from pathlib import Path
import shutil
from ultralytics import YOLO

class YoloTraining:
    def __init__(self, base_path, original_yaml='coco.yaml', temp_yaml='temp_coco.yaml'):
        self.base_path = base_path
        self.original_yaml = original_yaml
        self.temp_yaml = temp_yaml
        self.image_list = 'temp_image_list.txt'
        self.validation_list = 'validation_image_list.txt'
        self.predictions_file = 'predictions.json'
        self.weights_dir = Path(base_path, 'runs/detect/weights')
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.last_weights_path = self.weights_dir / 'last.pt'
        self.config = self.load_yaml(Path(base_path, original_yaml))
        
    def load_yaml(self, filepath):
        with open(filepath, 'r') as file:
            return yaml.safe_load(file)

    def save_yaml(self, data, filepath):
        with open(filepath, 'w') as file:
            yaml.safe_dump(data, file)

    def get_image_paths(self, directory):
        return [os.path.join(directory, img) for img in os.listdir(directory) if img.endswith('.jpg')]

    def save_image_list(self, image_paths, filepath):
        with open(filepath, 'w') as file:
            for path in image_paths:
                file.write(f"{path}\n")

    def select_images(self, image_paths, num_images=1000, seed=42):
        random.seed(seed)
        return random.sample(image_paths, num_images)

    def process_predictions(self, prediction_file_path, image_paths):
        with open(prediction_file_path, 'r') as file:
            predictions_data = json.load(file)

        no_detections, confidence_data = [], []

        for item in predictions_data:
            predictions = item['predictions']
            image_index = item['image_index']
            if not predictions:
                no_detections.append(image_index)
            else:
                average_confidence = np.mean([pred[4] for pred in predictions])
                confidence_data.append((image_index, average_confidence))

        selected_indices = self.select_for_retraining(no_detections, confidence_data, len(image_paths))
        selected_image_paths = [image_paths[i] for i in selected_indices]
        remaining_images = [img for i, img in enumerate(image_paths) if i not in selected_indices]
        return selected_image_paths, remaining_images

    def select_for_retraining(self, no_detections, confidence_data, total_images):
        if len(no_detections) >= 1000:
            return no_detections[:1000]
        else:
            confidence_data.sort(key=lambda x: x[1])
            samples_needed = 1000 - len(no_detections)
            return no_detections + [x[0] for x in confidence_data[:samples_needed]]

    def run_training_cycles(self, num_cycles=9, epochs=3, imgsz=224, batch=16, lr0=0.01):
        all_images = self.get_image_paths(self.config['train'])
        selected_images = self.select_images(all_images)
        remaining_images = list(set(all_images) - set(selected_images))

        for cycle in range(num_cycles):
            print(f"Cycle {cycle + 1}/{num_cycles}")
            if cycle != 0:
                selected_images, remaining_images = self.process_predictions(self.predictions_file, remaining_images)

            self.save_image_list(selected_images, Path(self.base_path, self.image_list))
            self.save_image_list(remaining_images, Path(self.base_path, self.validation_list))

            self.config['train'] = os.path.join(self.base_path, self.image_list)
            self.config['val'] = os.path.join(self.base_path, self.validation_list) if cycle < num_cycles - 1 else None
            self.save_yaml(self.config, Path(self.base_path, self.temp_yaml))

            model = self.train_model(Path(self.base_path, self.temp_yaml), epochs, imgsz, batch, lr0, val=(cycle < num_cycles - 1))
            self.update_weights()

        return model.val()

    def train_model(self, config_path, epochs, imgsz, batch, lr0, val=True):
        model = YOLO("yolov8n.yaml")
        model.train(data=config_path, epochs=epochs, imgsz=imgsz, batch=batch, lr0=lr0, val=val)
        return model

    def update_weights(self):
        latest_train_dir = max(Path(self.base_path, 'runs/detect').glob('train*'), key=os.path.getmtime)
        latest_weights_path = latest_train_dir / 'weights/last.pt'
        shutil.copy(latest_weights_path, self.last_weights_path)

# 使用
trainer = YoloTraining('/home/cyy/code/yolov8/user_yaml')
metrics = trainer.run_training_cycles()
import yaml
import random
import os
import json
import numpy as np
from pathlib import Path
import shutil
from ultralytics import YOLO

class YoloTraining:
    def __init__(self, base_path, original_yaml='coco.yaml', temp_yaml='temp_coco.yaml'):
        self.base_path = base_path
        self.original_yaml = original_yaml
        self.temp_yaml = temp_yaml
        self.image_list = 'temp_image_list.txt'
        self.validation_list = 'validation_image_list.txt'
        self.predictions_file = 'predictions.json'
        self.weights_dir = Path(base_path, 'runs/detect/weights')
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.last_weights_path = self.weights_dir / 'last.pt'
        self.config = self.load_yaml(Path(base_path, original_yaml))
        
    def load_yaml(self, filepath):
        with open(filepath, 'r') as file:
            return yaml.safe_load(file)

    def save_yaml(self, data, filepath):
        with open(filepath, 'w') as file:
            yaml.safe_dump(data, file)

    def get_image_paths(self, directory):
        return [os.path.join(directory, img) for img in os.listdir(directory) if img.endswith('.jpg')]

    def save_image_list(self, image_paths, filepath):
        with open(filepath, 'w') as file:
            for path in image_paths:
                file.write(f"{path}\n")

    def select_images(self, image_paths, num_images=1000, seed=42):
        random.seed(seed)
        return random.sample(image_paths, num_images)

    def process_predictions(self, prediction_file_path, image_paths):
        with open(prediction_file_path, 'r') as file:
            predictions_data = json.load(file)

        no_detections, confidence_data = [], []

        for item in predictions_data:
            predictions = item['predictions']
            image_index = item['image_index']
            if not predictions:
                no_detections.append(image_index)
            else:
                average_confidence = np.mean([pred[4] for pred in predictions])
                confidence_data.append((image_index, average_confidence))

        selected_indices = self.select_for_retraining(no_detections, confidence_data, len(image_paths))
        selected_image_paths = [image_paths[i] for i in selected_indices]
        remaining_images = [img for i, img in enumerate(image_paths) if i not in selected_indices]
        return selected_image_paths, remaining_images

    def select_for_retraining(self, no_detections, confidence_data, total_images):
        if len(no_detections) >= 1000:
            return no_detections[:1000]
        else:
            confidence_data.sort(key=lambda x: x[1])
            samples_needed = 1000 - len(no_detections)
            return no_detections + [x[0] for x in confidence_data[:samples_needed]]

    def run_training_cycles(self, num_cycles=9, epochs=3, imgsz=224, batch=16, lr0=0.01):
        all_images = self.get_image_paths(self.config['train'])
        selected_images = self.select_images(all_images)
        remaining_images = list(set(all_images) - set(selected_images))

        for cycle in range(num_cycles):
            print(f"Cycle {cycle + 1}/{num_cycles}")
            if cycle != 0:
                selected_images, remaining_images = self.process_predictions(self.predictions_file, remaining_images)

            self.save_image_list(selected_images, Path(self.base_path, self.image_list))
            self.save_image_list(remaining_images, Path(self.base_path, self.validation_list))

            self.config['train'] = os.path.join(self.base_path, self.image_list)
            self.config['val'] = os.path.join(self.base_path, self.validation_list) if cycle < num_cycles - 1 else None
            self.save_yaml(self.config, Path(self.base_path, self.temp_yaml))

            model = self.train_model(Path(self.base_path, self.temp_yaml), epochs, imgsz, batch, lr0, val=True)
            self.update_weights()

        return model.val()

    def train_model(self, config_path, epochs, imgsz, batch, lr0, val=True):
        model = YOLO("yolov8n.yaml")
        model.train(data=config_path, epochs=epochs, imgsz=imgsz, batch=batch, lr0=lr0, val=val)
        return model

    def update_weights(self):
        latest_train_dir = max(Path(self.base_path, 'runs/detect').glob('train*'), key=os.path.getmtime)
        latest_weights_path = latest_train_dir / 'weights/last.pt'
        shutil.copy(latest_weights_path, self.last_weights_path)

# 使用
trainer = YoloTraining('/home/cyy/code/yolov8/user_yaml')
metrics = trainer.run_training_cycles()
