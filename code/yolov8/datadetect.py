import os
import shutil
import random
from ultralytics import YOLO

# 定义用于生成训练和验证图片子集的函数
def generate_image_subsets(root_dir, train_dir, val_dir, selected_images, increment):
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    all_images = os.listdir(root_dir)
    unselected_images = [img for img in all_images if img not in selected_images]
    new_selected_images = random.sample(unselected_images, min(len(unselected_images), increment))
    selected_images.update(new_selected_images)

    for img in new_selected_images:
        shutil.copy(os.path.join(root_dir, img), train_dir)

    for img in os.listdir(val_dir):
        os.remove(os.path.join(val_dir, img))
    for img in unselected_images:
        if img not in selected_images:
            shutil.copy(os.path.join(root_dir, img), val_dir)

    return selected_images

# 初始化参数
root_dir = '/home/cyy/code/datasets/coco128/images/train2017'  # 原始图片目录
train_dir = '/home/cyy/code/train'  # 训练图片目录
val_dir = '/home/cyy/code/val'  # 验证图片目录
selected_images = set()  # 已选择图片的集合
increment = 4  # 每次想要增加的图片数量
epochs_per_increment = 10  # 每次增量训练的epoch数
total_images = 128  # 总图片数量


# 加载模型
model = YOLO("yolov8n.yaml").load("yolov8n.pt")

# 进行增量训练
while len(selected_images) < total_images:
    # 更新图片子集
    selected_images = generate_image_subsets(root_dir, train_dir, val_dir, selected_images, increment)

    # 使用更新后的训练和验证目录进行训练
    # 注意：这里假设train_dir和val_dir对应于coco128.yaml中的train和val路径
    # 如果使用不同的路径，需要相应地更新coco128.yaml文件或创建一个新的配置文件
    results = model.train(data='/home/cyy/code/yolov8/user_yaml/coco128.yaml', epochs=epochs_per_increment, imgsz=128, save = False,batch = 4,cache =False)