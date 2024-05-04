import os
import shutil
import random

# #此代码的作用是从coco2017数据集选取10000张样本来当作训练集


# # 设置原始图片文件夹路径
# src_directory = '/home/cyy/code/datasets/coco/images/train2017'
# # 设置目标图片文件夹路径
# dest_directory = '/home/cyy/code/datasets/coco_hhh/images/train'

# # 确保目标文件夹存在
# if not os.path.exists(dest_directory):
#     os.makedirs(dest_directory)

# # 获取所有图片的文件名
# all_files = [f for f in os.listdir(src_directory) if os.path.isfile(os.path.join(src_directory, f))]
# # 随机选择10000张图片，因为用于测试，所以选择10000张
# selected_files = random.sample(all_files, 10000)

# # 复制图片
# for file_name in selected_files:
#     shutil.copy(os.path.join(src_directory, file_name), os.path.join(dest_directory, file_name))

# print(f"已成功复制{len(selected_files)}张图片到{dest_directory}")
