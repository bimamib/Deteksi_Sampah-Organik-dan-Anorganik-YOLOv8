from ultralytics import YOLO
import numpy as np

model = YOLO(r"C:\Users\bimap\OneDrive\Documents\Yolov8\best.pt")  # load a custom model

results = model(r"C:\Users\bimap\OneDrive\Documents\Yolov8\kertass.jpeg")  # predict on an image

names_dict = results[0].names
probs = results[0].probs.data.tolist()

max_value = probs[0]
max_index = 0

for i in range(1, len(probs)):
    # Compare the current element value with the stored maximum value
    if probs[i] > max_value:
        # If the current element value is greater, update the maximum value and index
        max_value = probs[i]
        max_index = i

print("KLASIFIKASI:", names_dict[np.argmax(probs)])
print("PROBABILITAS:", max_value)
