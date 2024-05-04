import random
from ultralytics import YOLO
import os
import shutil
from pathlib import Path

fixed_weights_dir = Path('/home/cyy/code/runs/detect/weights')

fixed_weights_dir.mkdir(parents= True, exist_ok=True)

fixed_last_weights_path = fixed_weights_dir / 'last.pt'

def update_datasets(train_dir, val_dir, all_images_dir, train_images):
    #清空当前的训练和验证目录
    for filename in os.listdir(train_dir):
        file_path = os.path.join(train_dir, filename)
        if os.path.isfile(file_path):
            os.unlink(file_path)
    
    for filenme in os.listdir(val_dir):
        file_path = os.path.join(val_dir, filenme)
        if os.path.isfile(file_path):
            os.unlink(file_path)
    
    #更新训练和验证集
    for img in train_images:
        shutil.copy2(os.path.join(all_images_dir, img), train_dir)
  
    for img in set(all_images) - set(train_images):
        shutil.copy2(os.path.join(all_images_dir, img), val_dir)

if __name__ == '__main__':
    data_yaml = '/home/cyy/code/yolov8/user_yaml/coco_for_al.yaml'
    all_images_dir = '/home/cyy/code/datasets/coco/images/train2017'
    train_dir = '/home/cyy/code/datasets/coco_for_al/images/train'
    val_dir = '/home/cyy/code/datasets/coco_for_al/images/val'
    epochs_per_cycle = 150
    images_per_cycle = 10000
    cycles = 9 
    weights_dirs = '/home/cyy/code/weights'
    all_images = os.listdir(all_images_dir)
    random.shuffle(all_images)
    remaining_images = all_images.copy()
    train_images = []

    model = YOLO('yolov8n.yaml')

    for cycle in range(cycles):

        

        selected_images = random.sample(remaining_images, images_per_cycle)
        train_images.extend(selected_images)
        remaining_images = [img for img in remaining_images if img not in selected_images]
        
        update_datasets(train_dir, val_dir, all_images_dir, train_images )

        print(f"Cycle {cycle + 1}/{cycles}: Training with {len(train_images)} images")
        if cycle == 0:
            model.train(data = data_yaml, epochs = epochs_per_cycle, imgsz = 640, batch = 16, lr0 = 0.01, cache = True)
        elif cycle != 0:
            model.train(data = data_yaml, model=(str(fixed_last_weights_path)),epochs = epochs_per_cycle, imgsz = 640, batch = 16, lr0 = 0.01, cache = True, resume = True)
        
        latest_train_dir = max(Path('/home/cyy/code/runs/detect').glob('train*'), key = os.path.getmtime)
        latest_weights_path = latest_train_dir /'weights/last.pt'
        shutil.copy(latest_weights_path,fixed_last_weights_path)