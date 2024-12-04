import os
import json
import numpy as np
import cv2

def read_yolo_label(file_path):
    labels = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            label = {
                "class": int(parts[0]),
                "bbox": list(map(float, parts[1:5]))
            }
            labels.append(label)
    return labels

def convert_bbox_yolo_to_corners(bbox, img_width, img_height):
    x_center, y_center, width, height = bbox
    xmin = (x_center - width / 2) * img_width
    xmax = (x_center + width / 2) * img_width
    ymin = (y_center - height / 2) * img_height
    ymax = (y_center + height / 2) * img_height
    return [xmin, ymin, xmax, ymax]

def compute_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area

def parse_json_results(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data['result']['detections'], data['result']['directory']

def get_image_size(image_path):
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    return width, height

def calculate_metrics(detections, images_dir, labels_dir, class_names):
    all_detections = []
    all_annotations = []
    ious = []
    image_set = set()
    no_detections_list = []

    for detection in detections:
        image_name = detection['image']
        image_path = os.path.join(images_dir, image_name)
        img_width, img_height = get_image_size(image_path)

        label_name = os.path.splitext(image_name)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_name)

        if not os.path.exists(label_path):
            print(f"Label file not found: {label_path}")
            continue

        true_labels = read_yolo_label(label_path)

        if detection['class_name'] == 'null':
            no_detections_list.append(detection)
            for label in true_labels:
                true_box = convert_bbox_yolo_to_corners(label['bbox'], img_width, img_height)
                all_annotations.append({
                    'image_id': image_name,
                    'class': class_names[label['class']],
                    'bbox': true_box,
                    'label_id': id(label)
                })
                ious.append(0.0)  # IoU is 0 for null detections
            continue

        detected_box = [
            detection['bounding_box']['xmin'],
            detection['bounding_box']['ymin'],
            detection['bounding_box']['xmax'],
            detection['bounding_box']['ymax']
        ]

        all_detections.append({
            'image_id': image_name,
            'class': detection['class_name'],
            'bbox': detected_box,
            'confidence': detection['confidence']
        })

        image_set.add(image_name)

        print(f"Image: {image_name}")
        print(
            f"Detected class: {detection['class_name']}, Bounding box: {detected_box}, Confidence: {detection['confidence']}")
        print(f"Image dimensions: width={img_width}, height={img_height}")

        for label in true_labels:
            true_box = convert_bbox_yolo_to_corners(label['bbox'], img_width, img_height)
            all_annotations.append({
                'image_id': image_name,
                'class': class_names[label['class']],
                'bbox': true_box,
                'label_id': id(label)
            })
            iou = compute_iou(detected_box, true_box)
            ious.append(iou)
            print(f"Label class: {class_names[label['class']]}, Bounding box: {true_box}, IoU: {iou}")

    for annotation in all_annotations:
        image_set.add(annotation['image_id'])

    total_images = len(image_set)

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for class_name in class_names:
        class_detections = [det for det in all_detections if det['class'] == class_name]
        class_annotations = [ann for ann in all_annotations if ann['class'] == class_name]

        if len(class_annotations) == 0:
            false_positives += len(class_detections)
            continue

        annotations_by_image = {}
        for ann in class_annotations:
            if ann['image_id'] not in annotations_by_image:
                annotations_by_image[ann['image_id']] = []
            annotations_by_image[ann['image_id']].append(ann)

        class_detections = sorted(class_detections, key=lambda x: x['confidence'], reverse=True)

        matched_annotations = set()

        for det in class_detections:
            image_id = det['image_id']
            if image_id in annotations_by_image:
                max_iou = 0
                matched_annotation = None

                for ann in annotations_by_image[image_id]:
                    iou = compute_iou(det['bbox'], ann['bbox'])
                    if iou > max_iou:
                        max_iou = iou
                        matched_annotation = ann

                if max_iou >= 0.5 and matched_annotation['label_id'] not in matched_annotations:
                    true_positives += 1
                    matched_annotations.add(matched_annotation['label_id'])
                else:
                    false_positives += 1
            else:
                false_positives += 1

        false_negatives += len(class_annotations) - len(matched_annotations)

    tp = true_positives
    fp = false_positives
    fn = false_negatives + len(no_detections_list)  # Add null detections to FN

    precision = tp / (tp + fp + np.finfo(np.float64).eps)
    recall = tp / (tp + fn + np.finfo(np.float64).eps)
    f1_score = 2 * (precision * recall) / (precision + recall + np.finfo(np.float64).eps)

    ap_sum = 0.0
    for t in np.linspace(0, 1, 11):
        precisions_at_t = [p for p, r in zip([precision], [recall]) if r >= t]
        if len(precisions_at_t) > 0:
            ap_sum += max(precisions_at_t)
    mAP = ap_sum / 11

    # Calculate True Negatives (TN)
    tn = total_images - (tp + fp + fn)
    if tn < 0: tn = 0  # Ensure TN is not negative

    # Add no detections at the end of the metrics output
    no_detections_count = len(no_detections_list)
    print(f"\nNo Detections for the following images ({no_detections_count}):")
    for no_detection in no_detections_list:
        print(no_detection['image'])

    return mAP, precision, recall, f1_score, np.mean(ious), tp, fp, fn, tn, total_images, no_detections_count

# Example usage
json_file = 'result/result_normal.json'
labels_dir = 'test/labels'
class_names = ['Anorganik', 'Organik']
detections, images_dir = parse_json_results(json_file)
mAP, precision, recall, f1_score, average_iou, tp, fp, fn, tn, total_images, no_detections_count = calculate_metrics(detections, images_dir, labels_dir, class_names)

print(f"Mean Average Precision (mAP): {mAP}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1_score}")
print(f"Average IoU: {average_iou}")
print(f"Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
print(f"Total Images: {total_images}")
print(f"No Detections Count: {no_detections_count}")
