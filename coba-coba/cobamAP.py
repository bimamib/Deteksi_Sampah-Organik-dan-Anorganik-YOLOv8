import numpy as np
from collections import defaultdict

def compute_iou(box1, box2):
    """
    Menghitung Intersection over Union (IoU) antara dua bounding box.
    box: [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    # Hitung koordinat persimpangan antara dua box
    x_intersect1 = max(x1, x3)
    y_intersect1 = max(y1, y3)
    x_intersect2 = min(x2, x4)
    y_intersect2 = min(y2, y4)

    # Hitung luas persimpangan
    intersect = max(0, x_intersect2 - x_intersect1 + 1) * max(0, y_intersect2 - y_intersect1 + 1)

    # Hitung luas gabungan
    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x4 - x3 + 1) * (y4 - y3 + 1)
    union = box1_area + box2_area - intersect

    # Hitung IoU
    iou = intersect / union

    return iou

def compute_map(detections, ground_truth, iou_thresh=0.5):
    """
    Menghitung mAP (mean Average Precision) untuk deteksi objek.
    detections: list of [class_id, confidence, box]
    ground_truth: list of [class_id, box]
    """
    # Inisialisasi variabel
    true_positives = defaultdict(int)
    false_positives = defaultdict(int)
    false_negatives = defaultdict(int)
    precisions = defaultdict(list)
    recalls = defaultdict(list)

    # Iterasi melalui setiap gambar
    for gt_boxes, dt_boxes in zip(ground_truth, detections):
        for class_id, gt_box in gt_boxes:
            found = False
            for dt_class_id, confidence, dt_box in dt_boxes:
                iou = compute_iou(gt_box, dt_box)
                if iou >= iou_thresh and dt_class_id == class_id:
                    found = True
                    true_positives[class_id] += 1
                elif iou >= iou_thresh and dt_class_id != class_id:
                    false_positives[class_id] += 1
            if not found:
                false_negatives[class_id] += 1

        # Hitung presisi dan recall untuk setiap kelas
        for class_id in set(true_positives.keys()).union(set(false_positives.keys())).union(set(false_negatives.keys())):
            tp = true_positives[class_id]
            fp = false_positives[class_id]
            fn = false_negatives[class_id]
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            precisions[class_id].append(precision)
            recalls[class_id].append(recall)

    # Hitung AP (Average Precision) untuk setiap kelas
    average_precisions = {}
    for class_id in set(precisions.keys()):
        sorted_precisions = sorted(precisions[class_id], reverse=True)
        sorted_recalls = sorted(recalls[class_id], reverse=True)
        ap = 0
        for i in range(len(sorted_precisions)):
            ap += sorted_precisions[i] * (sorted_recalls[i] - sorted_recalls[i - 1] if i > 0 else sorted_recalls[i])
        average_precisions[class_id] = ap

    # Hitung mAP (mean Average Precision)
    map = sum(average_precisions.values()) / len(average_precisions)

    return map