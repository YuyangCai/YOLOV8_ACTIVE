import json
import yaml
from ultralytics import YOLO
import numpy as np
# Load the data from the JSON file
with open('predictions.json', 'r') as file:
    data = json.load(file)

# Initialize lists to store images with no detections and their confidence data
no_detections = []
confidence_data = []

# Process each image in the dataset
for item in data:
    predictions = item['predictions']
    image_index = item['image_index']
    
    if not predictions:  # Check if there are no detections
        no_detections.append(image_index)
    else:
        # Calculate average confidence for images with detections
        # average_confidence = sum(pred[4] for pred in predictions) / len(predictions)
        # confidence_data.append((image_index, average_confidence))
        predictions_array = np.array(predictions)
        average_confidence = np.mean(predictions_array[:, 4])   #改为使用数组的方式计算，二者计算时间效率差不多
        confidence_data.append((image_index, average_confidence))

# Sort the confidence data by average confidence (ascending order)
confidence_data.sort(key=lambda x: x[1])

# Determine the total number of samples needed
samples_needed = 1000 - len(no_detections)  #排除空样本后在剩下的样本中选取出置信度最低的

# Select the lowest confidence samples if there aren't enough no detection samples
selected_samples = no_detections + [x[0] for x in confidence_data[:samples_needed]]  #将没有预测到的样本和置信度最低的样本数相加

# Print the results
print(f"Total samples selected: {len(selected_samples)}")
print("Sample indices selected:", selected_samples)

