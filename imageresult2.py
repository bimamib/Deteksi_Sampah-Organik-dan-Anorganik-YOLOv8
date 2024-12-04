import os
import json
import cv2

# Path to the JSON file
json_file_path = 'result_darkness/result_darkness1.json'

# Load JSON data from file
with open(json_file_path, 'r') as file:
    detections_data = json.load(file)

# Function to change the directory path from images to labels and change extension to .txt
def change_directory_and_extension(image_path):
    label_path = image_path.replace(r'test_darkness\images', r'test_darkness\labels')
    label_path = os.path.splitext(label_path)[0] + '.txt'
    return label_path

# Create result directory if not exists
result_dir = "result_darkness"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# Function to read label data from YOLO format file
def read_yolo_labels(label_file_path, image_width, image_height):
    labels = []
    class_mapping = {'0': 'Anorganik', '1': 'Organik'}
    with open(label_file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = class_mapping.get(parts[0], 'Unknown')
                x_center = float(parts[1]) * image_width
                y_center = float(parts[2]) * image_height
                width = float(parts[3]) * image_width
                height = float(parts[4]) * image_height
                xmin = int(x_center - width / 2)
                ymin = int(y_center - height / 2)
                xmax = int(x_center + width / 2)
                ymax = int(y_center + height / 2)
                labels.append({
                    'class_name': class_id,
                    'bounding_box': {
                        'xmin': xmin,
                        'ymin': ymin,
                        'xmax': xmax,
                        'ymax': ymax
                    }
                })
    return labels

# Function to draw bounding boxes and save the image
def draw_bounding_boxes(image_path, detection, labels, result_dir):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image {image_path} not found.")
        return

    image_height, image_width, _ = image.shape

    # Draw bounding box from JSON in red
    bbox = detection['bounding_box']
    xmin, ymin, xmax, ymax = int(bbox['xmin']), int(bbox['ymin']), int(bbox['xmax']), int(bbox['ymax'])
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)  # Red color

    # Add class name and confidence with black background
    label_text = f"{detection['class_name']} ({detection['confidence']:.2f})"
    text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
    cv2.rectangle(image, (xmax - text_size[0] - 5, ymax + 5), (xmax + 5, ymax + text_size[1] + 15), (0, 0, 0), -1)
    cv2.putText(image, label_text, (xmax - text_size[0], ymax + text_size[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Draw bounding box from labels in blue
    for label in labels:
        label_bbox = label['bounding_box']
        xmin_label, ymin_label, xmax_label, ymax_label = int(label_bbox['xmin']), int(label_bbox['ymin']), int(label_bbox['xmax']), int(label_bbox['ymax'])
        cv2.rectangle(image, (xmin_label, ymin_label), (xmax_label, ymax_label), (255, 0, 0), 2)  # Blue color

        # Add class name for label with black background
        label_text = f"Asli: {label['class_name']}"
        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(image, (xmin_label - 5, ymax_label + 5), (xmin_label + text_size[0] + 5, ymax_label + text_size[1] + 15), (0, 0, 0), -1)
        cv2.putText(image, label_text, (xmin_label, ymax_label + text_size[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Save the result image
    result_image_path = os.path.join(result_dir, os.path.basename(image_path))
    cv2.imwrite(result_image_path, image)

# Process each detection
for detection in detections_data['result']['detections']:
    # Find the corresponding label for the image
    image_name = detection['image']
    original_image_path = os.path.join(detections_data['result']['directory'], image_name)
    label_file_path = change_directory_and_extension(original_image_path)

    # Load label data for the image
    try:
        image = cv2.imread(original_image_path)
        if image is not None:
            image_height, image_width, _ = image.shape
            labels = read_yolo_labels(label_file_path, image_width, image_height)
            draw_bounding_boxes(original_image_path, detection, labels, result_dir)
    except FileNotFoundError:
        print(f"Label file {label_file_path} not found. Skipping.")

print("Bounding boxes drawn and images saved to the result directory.")
