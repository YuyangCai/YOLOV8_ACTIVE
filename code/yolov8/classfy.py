from ultralytics import YOLO
 
 
# # 加载模型
# model = YOLO('yolov8l-cls.yaml')  # 从YAML构建并转移权重
 
# if __name__ == '__main__':
#     # 训练模型
#     results = model.train(data='/home/cyy/code/datasets/classified_coco', epochs=300, imgsz=224, batch = 32, lr0 = 0.02 ,
#                           device = 0,cache = True,  save = True)

# '''预测'''
#     # 预测模型


# Load a model
model = YOLO('/home/cyy/code/runs/classify/train32/weights/best.pt')  # load an official model
# model = YOLO('/home/cyy/code/datasets/classified_coco/train/apple/000000000670.jpg')  # load a custom model

# Predict with the model
results = model.predict('/home/cyy/code/datasets/coco/images/test2017',save = True)  # predict on an image