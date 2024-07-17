import os
import json
import shutil
from pascal_voc_writer import Writer
from pycocotools.coco import COCO

'''生成voc数据集所需的特定的xml格式标注,同时将val和train图片全放入JPEGImages下'''
def gen_annotations_images(coco_json_path, images_dir, output_dir_annotations, output_dir_images ):
    # 加载COCO数据集
    coco = COCO(coco_json_path)

    # # 创建输出目录
    os.makedirs(output_dir_annotations, exist_ok=True)

    # 读取COCO的所有类别
    categories = coco.loadCats(coco.getCatIds())
    category_id_to_name = {category['id']: category['name'] for category in categories}

    # 遍历每张图片并转换标注
    for img_id in coco.getImgIds():
        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info['file_name']
        width = img_info['width']
        height = img_info['height']

        # 创建VOC格式的XML文件
        writer = Writer(file_name, width, height)

        # 获取图片中的所有标注
        annotation_ids = coco.getAnnIds(imgIds=img_id)
        annotations = coco.loadAnns(annotation_ids)

        for annotation in annotations:
            category_id = annotation['category_id']
            category_name = category_id_to_name[category_id]
            bbox = annotation['bbox']

            # COCO的bbox格式为[x, y, width, height]
            x_min = int(bbox[0])
            y_min = int(bbox[1])
            x_max = int(bbox[0] + bbox[2])
            y_max = int(bbox[1] + bbox[3])

            writer.addObject(category_name, x_min, y_min, x_max, y_max)

        # 保存XML文件
        xml_path = os.path.join(output_dir_annotations, os.path.splitext(file_name)[0] + '.xml')
        writer.save(xml_path)

        #  复制图片到输出目录
        shutil.copy(os.path.join(images_dir, file_name), os.path.join(output_dir_images, file_name))    

'''生成训练voc数据集所需的.txt文件索引'''
def gen_txt(images_dir, output_file):
    # 获取文件夹下所有文件名
    filenames = os.listdir(images_dir)
    
    # 过滤出图片文件（假设图片扩展名为jpg或png）
    image_filenames = [os.path.splitext(filename)[0] for filename in filenames if filename.endswith(('.jpg', '.png'))]
    
    # 按字母顺序排序文件名
    image_filenames.sort()
    
    # 将文件名写入txt文件
    with open(output_file, 'w') as f:
        for filename in image_filenames:
            f.write(filename + '\n')

if __name__ == '__main__':
    image_dir_train = '/home/data/dataset_AL_coco/coco/images/train2017'
    image_dir_val   = '/home/data/dataset_AL_coco/coco/images/val2017'
    coco_json_path_train = '/home/cyy/instances_train2017.json'
    coco_json_path_val   = '/home/cyy/instances_val2017.json'

    output_annotation = '/home/cyy/test_for_datasets/coco2voc/VOCdevkit/VOC2012/Annotations'
    output_image      = '/home/cyy/test_for_datasets/coco2voc/VOCdevkit/VOC2012/JPEGImages'

    output_file_train = '/home/cyy/test_for_datasets/coco2voc/VOCdevkit/VOC2012/ImageSets/Main/train.txt'
    output_file_val   = '/home/cyy/test_for_datasets/coco2voc/VOCdevkit/VOC2012/ImageSets/Main/val.txt'

    gen_annotations_images(coco_json_path_train, image_dir_train, output_annotation, output_image)
    gen_annotations_images(coco_json_path_val  , image_dir_val  , output_annotation, output_image)

    gen_txt(image_dir_train, output_file_train)
    gen_txt(image_dir_val, output_file_val)