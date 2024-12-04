import json
import os
import cv2
import time
import torch
from ultralytics import YOLO

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

        # Inference
        start_inference = time.time()
        results = model.predict(image, max_det=1)
        end_inference = time.time()

        # Postprocess
        start_postprocess = time.time()
        boxes = results[0].boxes.xyxyn.cpu().numpy() * [image.shape[1], image.shape[0], image.shape[1], image.shape[0]]
        confidences = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        class_names = [model.names[cls_id] for cls_id in class_ids]
        speed = results[0].speed
        end_postprocess = time.time()

        # Extract the best confidence score and the corresponding bounding box
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
                    "preprocess": (end_preprocess - start_preprocess) * 1000, # for processing time in milliseconds
                    "inference": (end_inference - start_inference) * 1000,
                    "postprocess": (end_postprocess - start_postprocess) * 1000
                }
            }

            # Add detection to results dictionary
            results_dict["result"]["detections"].append(detection)
        else:
            # If no detections, add the image with class name 'null'
            detection = {
                "image": image_file,
                "class_name": "null"
            }
            results_dict["result"]["detections"].append(detection)
            print(f"No detections for {image_file}.")

    # Save results to JSON file in the provided result directory
    result_path = os.path.join(result_directory, "result_motion_blur.json")
    with open(result_path, 'w') as outfile:
        json.dump(results_dict, outfile, indent=4)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Detect objects in images using YOLOv8 and save results to a JSON file.')
    parser.add_argument('image_directory', type=str, help='Directory containing images to be processed.')
    parser.add_argument('result_directory', type=str, help='Directory to save the results.')

    args = parser.parse_args()
    main(args.image_directory, args.result_directory)
