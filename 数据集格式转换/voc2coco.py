import os
import shutil
import xml.etree.ElementTree as ET

# VOC 类别
voc_classes = {
    "aeroplane": 1,
    "bicycle": 2,
    "bird": 3,
    "boat": 4,
    "bottle": 5,
    "bus": 6,
    "car": 7,
    "cat": 8,
    "chair": 9,
    "cow": 10,
    "diningtable": 11,
    "dog": 12,
    "horse": 13,
    "motorbike": 14,
    "person": 15,
    "pottedplant": 16,
    "sheep": 17,
    "sofa": 18,
    "train": 19,
    "tvmonitor": 20
}

'''将voc数据集转化成coco格式并放入指定文件夹中,voc数据集格式与coco数据集格式有些许不同,数据集的划分通过txt文件'''
def copy_images(voc_images_path, txt_path , output_images_path):
    os.makedirs(output_images_path, exist_ok=True)
    with open(txt_path, 'r') as file:
         image_ids = file.read().splitlines()
    for image_id in image_ids:
        src_image_path  = os.path.join(voc_images_path,    f"{image_id}.jpg")
        dest_image_path = os.path.join(output_images_path, f"{image_id}.jpg")

        if os.path.exists(src_image_path):
            shutil.copy(src_image_path, dest_image_path)
            print(f"Copied {src_image_path} to {dest_image_path}")
        else:
            print(f"Image {src_image_path} does not exist")

'''将voc数据集.xml格式的annotations转化成与coco数据集相同的格式'''
def gen_labels(voc_annotations_path, txt_path, labels_output_dir):
    with open(txt_path, 'r') as file:
        image_ids = file.read().splitlines()
    for image_id in image_ids:
        # 解析 XML 文件
        xml_path = os.path.join(voc_annotations_path, f"{image_id}.xml")
        tree = ET.parse(xml_path)
        root = tree.getroot()

    # 获取图像尺寸
        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)

    # 标签文件路径
        label_file_path = os.path.join(labels_output_dir, f"{image_id}.txt")
        with open(label_file_path, 'w') as label_file:
            # 获取标注信息
            for obj in root.findall("object"):
                class_name = obj.find("name").text
                if class_name not in voc_classes:
                    continue
                class_id = voc_classes[class_name] - 1  # YOLO 类别从 0 开始

                bndbox = obj.find("bndbox")
                xmin = int(bndbox.find("xmin").text) - 1
                ymin = int(bndbox.find("ymin").text) - 1
                xmax = int(bndbox.find("xmax").text)
                ymax = int(bndbox.find("ymax").text)
                width_box = xmax - xmin
                height_box = ymax - ymin

                # 计算中心点和宽高的归一化坐标
                x_center = (xmin + width_box / 2) / width
                y_center = (ymin + height_box / 2) / height
                norm_width = width_box / width
                norm_height = height_box / height

                # 写入标签文件
                label_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n")


#原始voc图片路径信息
voc_images_path = "/home/cyy/VOCdevkit/VOC2012/JPEGImages"
train_txt_path  = "/home/cyy/VOCdevkit/VOC2012/ImageSets/Main/train.txt"
val_txt_path    = "/home/cyy/VOCdevkit/VOC2012/ImageSets/Main/val.txt"

#原始voc图片标注信息
voc_annotations_path = "/home/cyy/VOCdevkit/VOC2012/Annotations"

#训练集和验证集图片输出路径
output_images_path_train = "/home/cyy/test_for_datasets/voc2coco/images/train2017"
output_images_path_val   = "/home/cyy/test_for_datasets/voc2coco/images/val2017"

#～标签输出路径
labels_output_dir_train = "/home/cyy/test_for_datasets/voc2coco/labels/train2017"
labels_output_dir_val   = "/home/cyy/test_for_datasets/voc2coco/labels/val2017"

copy_images(voc_images_path, train_txt_path, output_images_path_train)
copy_images(voc_images_path, val_txt_path  , output_images_path_val)

gen_labels(voc_annotations_path, train_txt_path, labels_output_dir_train)
gen_labels(voc_annotations_path, val_txt_path, labels_output_dir_val)