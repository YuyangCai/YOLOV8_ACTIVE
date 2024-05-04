from pathlib import Path

def count_images_in_directory(directory):
    # 创建一个Path对象
    path = Path(directory)
    # 使用glob方法找到所有jpg, png, jpeg格式的图片
    image_files = list(path.glob('**/*.jpg')) + list(path.glob('**/*.png')) + list(path.glob('**/*.jpeg'))
    # 返回找到的图片数量
    return len(image_files)

# 使用函数
directory = '/home/cyy/code/samples/test6'  # 请替换为你的目标文件夹路径
image_count = count_images_in_directory(directory)
print(f"There are {image_count} images in the directory.")