import cv2
import os
import torch
from ultralytics import YOLO

# Initialize the model
model = YOLO('best3.pt')
results = model.predict(source="target")

# Create 'result' directory if it doesn't exist
if not os.path.exists('result'):
    os.makedirs('result')

for result in results:
    # Read the image
    image_path = result.path
    image = cv2.imread(image_path)

    # Get the bounding boxes (normalized), confidence scores, and speed
    boxes = result.boxes.xyxyn.cpu().numpy() * [image.shape[1], image.shape[0], image.shape[1], image.shape[0]]
    confidences = result.boxes.conf.cpu().numpy()
    speed = result.speed

    # Draw bounding boxes, confidence scores, and speed on the image
    for box, conf in zip(boxes, confidences):
        x1, y1, x2, y2 = map(int, box)
        label = f"Conf: {conf:.2f}"
        print(f"Conf: {conf:.2f}")
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0), 2)

    speed_text = f"Preprocess: {speed['preprocess']:.2f}ms, Inference: {speed['inference']:.2f}ms, Postprocess: {speed['postprocess']:.2f}ms"
    cv2.putText(image, speed_text, (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 2)

    # Save the image in the 'result' folder
    save_path = os.path.join('result', os.path.basename(image_path))
    cv2.imwrite(save_path, image)

print("Processing complete. Images saved in the 'result' folder.")
