import json
import os
import cv2
import time
import torch
import pandas as pd
from ultralytics import YOLO

def create_excel_from_detections(results_dict, result_path):
    # Create lists to hold the data
    data = []
    for i, detection in enumerate(results_dict["result"]["detections"], 1):
        try:
            row = [
                i,
                detection["image"],
                detection.get("real_class_name", None),
                detection.get("class_name", None),
                detection["verdict"],
                detection.get("confidence", None),
                detection.get("IoU", None),
                detection["time"].get("inference", None)
            ]
        except KeyError as e:
            print(f"Missing key {e} in detection: {detection}")
            continue
        data.append(row)

    # Create a DataFrame
    df = pd.DataFrame(data,
                      columns=["No", "Image Name", "Real Class", "Detection Class", "Verdict", "Confidence", "IoU", "Inference Time"])

    # Calculate statistics
    avg_iou = df["IoU"].mean()
    avg_inference_time = df["Inference Time"].mean()

    # Confusion matrix counts
    tp = len(df[df["Verdict"] == "TP"])
    fn = len(df[df["Verdict"] == "FN"])
    fp = len(df[df["Verdict"] == "FP"])
    tn = len(df[df["Verdict"] == "TN"])

    # Precision, Recall, F1 Score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # mAP (mean Average Precision) calculation can be complex and requires AP per class, simplifying here
    mAP = precision  # Simplified assumption

    # Add statistics to the DataFrame
    stats_data = {
        "Metric": ["Average IoU", "Average Inference Time", "TP", "FP", "FN", "TN", "Precision", "Recall", "F1 Score", "mAP"],
        "Value": [avg_iou, avg_inference_time, tp, fp, fn, tn, precision, recall, f1_score, mAP]
    }
    stats_df = pd.DataFrame(stats_data)

    # Write to Excel
    with pd.ExcelWriter(result_path) as writer:
        df.to_excel(writer, sheet_name="Detections", index=False)
        stats_df.to_excel(writer, sheet_name="Statistics", index=False)

def change_directory_and_extension(image_path):
    label_path = image_path.replace(r'\images', r'\labels')
    label_path = os.path.splitext(label_path)[0] + '.txt'
    return label_path

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

def compute_iou(box1, box2):
    x1 = max(box1['xmin'], box2['xmin'])
    y1 = max(box1['ymin'], box2['ymin'])
    x2 = min(box1['xmax'], box2['xmax'])
    y2 = max(box1['ymax'], box2['ymax'])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1['xmax'] - box1['xmin']) * (box1['ymax'] - box1['ymin'])
    box2_area = (box2['xmax'] - box2['xmin']) * (box2['ymax'] - box2['ymin'])
    union = box1_area + box2_area - intersection

    iou = intersection / union if union != 0 else 0
    return iou

def compute_verdict(iou, threshold=0.5):
    if iou >= threshold:
        return 'TP'
    elif iou < threshold and iou > 0:
        return 'FP'
    else:
        return 'FN'

def main(image_directory, result_directory):
    # Initialize the model
    model = YOLO('best3.pt')

    # Initialize results dictionary
    results_dict = {
        "result": {
            "directory": image_directory,
            "detections": []
        }
    }

    # Adjust path to point to /images/ sub-directory
    image_directory = os.path.join(image_directory)

    # Get list of images in the directory
    image_files = [f for f in os.listdir(image_directory) if
                   os.path.isfile(os.path.join(image_directory, f)) and f.endswith(('.jpg', '.jpeg', '.png'))]

    # Loop over each image and process
    for image_file in image_files:
        print(f"Processing: {image_file}")

        # Preprocess
        start_preprocess = time.time()
        image_path = os.path.join(image_directory, image_file)
        image = cv2.imread(image_path)
        end_preprocess = time.time()

        # Read ground truth labels
        label_file_path = change_directory_and_extension(image_path)
        real_labels = read_yolo_labels(label_file_path, image.shape[1], image.shape[0])
        real_label = real_labels[0] if real_labels else None

        # Inference
        start_inference = time.time()
        try:
            results = model.predict(image, max_det=1)  # Limit detections to 1 to ensure only the best detection is considered
            end_inference = time.time()
        except Exception as e:
            print(f"Inference error: {e}")
            results = []

        # Postprocess
        start_postprocess = time.time()
        if results:
            boxes = results[0].boxes.xyxyn.cpu().numpy() * [image.shape[1], image.shape[0], image.shape[1], image.shape[0]]
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            class_names = [model.names[cls_id] for cls_id in class_ids]
            end_postprocess = time.time()

            # Ensure only the best detection is recorded
            if confidences.size > 0:  # Check if there are any detections
                best_conf_index = confidences.argmax()
                best_conf = confidences[best_conf_index]
                best_box = boxes[best_conf_index]
                best_class_id = class_ids[best_conf_index]
                best_class_name = class_names[best_conf_index]

                # Create detection dictionary
                detection = {
                    "image": image_file,
                    "class_name": best_class_name,
                    "bounding_box": {
                        "xmin": best_box[0].item(),
                        "ymin": best_box[1].item(),
                        "xmax": best_box[2].item(),
                        "ymax": best_box[3].item()
                    },
                    "confidence": best_conf.item(),
                    "time": {
                        "preprocess": (end_preprocess - start_preprocess) * 1000,  # processing time in milliseconds
                        "inference": (end_inference - start_inference) * 1000,
                        "postprocess": (end_postprocess - start_postprocess) * 1000
                    }
                }

                if real_label:
                    iou = compute_iou(detection['bounding_box'], real_label['bounding_box'])
                    verdict = compute_verdict(iou)
                    detection.update({
                        "real_class_name": real_label['class_name'],
                        "real_bounding_box": real_label['bounding_box'],
                        "IoU": iou,
                        "verdict": verdict
                    })
                else:
                    detection.update({
                        "real_class_name": "null",
                        "real_bounding_box": {},
                        "IoU": 0,
                        "verdict": "FN"
                    })

                # Add detection to results dictionary
                results_dict["result"]["detections"].append(detection)
            else:
                # If no detections, add the image with class name 'null'
                detection = {
                    "image": image_file,
                    "class_name": "null",
                    "confidence": 0,
                    "time": {
                        "preprocess": (end_preprocess - start_preprocess) * 1000,
                        "inference": (end_inference - start_inference) * 1000,
                        "postprocess": (end_postprocess - start_postprocess) * 1000
                    },
                    "real_class_name": real_label['class_name'] if real_label else "null",
                    "real_bounding_box": real_label['bounding_box'] if real_label else {},
                    "IoU": 0,
                    "verdict": "TN" if not real_label else "FN"
                }
                results_dict["result"]["detections"].append(detection)
                print(f"No detections for {image_file}.")
        else:
            # If inference error or no results
            end_inference = time.time()
            detection = {
                "image": image_file,
                "class_name": "null",
                "confidence": 0,
                "time": {
                    "preprocess": (end_preprocess - start_preprocess) * 1000,
                    "inference": (end_inference - start_inference) * 1000,
                    "postprocess": None
                },
                "real_class_name": real_label['class_name'] if real_label else "null",
                "real_bounding_box": real_label['bounding_box'] if real_label else {},
                "IoU": 0,
                "verdict": "TN" if not real_label else "FN"
            }
            results_dict["result"]["detections"].append(detection)
            print(f"No detections for {image_file}.")

    # Save results to JSON file in the provided result directory
    result_path = os.path.join(result_directory, "result_motion_blur2.json")
    with open(result_path, 'w') as outfile:
        json.dump(results_dict, outfile, indent=4)

    # Use the previous results dictionary to create an Excel file
    excel_result_path = os.path.join(result_directory, "result_motion_blur2.xlsx")
    create_excel_from_detections(results_dict, excel_result_path)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Detect objects in images using YOLOv8 and save results to a JSON file.')
    parser.add_argument('image_directory', type=str, help='Directory containing images to be processed.')
    parser.add_argument('result_directory', type=str, help='Directory to save the results.')

    args = parser.parse_args()
    main(args.image_directory, args.result_directory)
