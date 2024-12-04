import ultralytics
from ultralytics import YOLO
ultralytics.checks()

import os
import sys

dataset = "dataset/data.yaml"

model = YOLO('yolov8n.pt')

result = model.train(data=dataset, epochs=30, imgsz=640)