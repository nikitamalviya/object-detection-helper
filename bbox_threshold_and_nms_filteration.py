'''
This moddule contains the source code to filter out the bounding boxes, to have a improved and best considered bboxes for detections using:
1. Threshold score(confidence score for detection of a bbox)
2. NMS score (the overlapping score between two the bounding boxes using IoU)
'''
import numpy as np, click

def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.
    Parameters:
        box1, box2: Lists or arrays representing bounding boxes [x1, y1, x2, y2].
    Returns:
        iou: Float representing the IoU between the two boxes.
    """
    # Calculate the intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1]) 
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    # Calculate the areas of the bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    # Calculate the union area
    union_area = box1_area + box2_area - intersection_area
    # Calculate the IoU
    iou = intersection_area / union_area if union_area != 0 else 0
    return iou

def apply_nms(boxes, scores, iou_threshold):
    """
    Apply Non-Maximum Suppression (NMS) to filter overlapping bounding boxes.
    Parameters:
        boxes: List of bounding boxes to apply NMS on.
        scores: List of scores associated with the bounding boxes.
        iou_threshold: Float value representing the IoU threshold for suppression.
    Returns:
        selected_indices: List of indices of the bounding boxes that are selected after NMS.
    """
    printFlag = False
    indices = list(range(len(boxes)))
    selected_indices = []
    while indices:
        current = indices[0]
        selected_indices.append(current)
        remaining_indices = []
        for i in indices[1:]:
            iou = calculate_iou(boxes[current], boxes[i])
            if iou <= iou_threshold:
                remaining_indices.append(i)
            else:
                if printFlag : click.secho(f"Eliminated box {i} due to IoU {iou:.2f} with box {current} index.", fg="red")
        indices = remaining_indices
    return selected_indices

def filter_boxes_with_nms(boxes, scores, threshold, iou_threshold):
    """
    Filter bounding boxes using a score threshold and apply Non-Maximum Suppression (NMS).
    Parameters:
        boxes: List of bounding boxes.
        scores: List of scores associated with the bounding boxes.
        threshold: Float value representing the minimum score threshold for filtering boxes.
        iou_threshold: Float value representing the IoU threshold for NMS.
    Returns:
        selected_filtered_indices: List of indices of the filtered and NMS-selected bounding boxes.
    """
    printFlag = False
    # Filter boxes based on the score threshold
    filtered_indices = [i for i in range(len(scores)) if scores[i] >= threshold]
    filtered_boxes = [boxes[i] for i in filtered_indices]
    filtered_scores = [scores[i] for i in filtered_indices]
    # Apply NMS on the filtered boxes
    selected_indices = apply_nms(filtered_boxes, filtered_scores, iou_threshold)
    # Map selected indices back to the original list of boxes
    selected_filtered_indices = [filtered_indices[i] for i in selected_indices]
    if printFlag : click.secho(f"Selected {len(selected_filtered_indices)} boxes after NMS", fg="green")
    return selected_filtered_indices

def process_class_annotations(class_name, boxes, scores, threshold, nms_threshold, printFlag):
    """
    Process bounding boxes for a specific class by filtering and applying NMS.
    Parameters:
        class_name: String representing the name of the class (e.g., 'human', 'head').
        boxes: List of bounding boxes for the class.
        scores: List of scores associated with the bounding boxes.
        threshold: Float value representing the score threshold for filtering boxes.
        nms_threshold: Float value representing the IoU threshold for NMS.
    Returns:
        nms_boxes: List of bounding boxes selected after NMS.
    """
    if len(boxes) > 0:
        # Filter boxes based on the score threshold
        filtered_indices = [i for i in range(len(scores)) if scores[i] >= threshold]
        filtered_boxes = [boxes[i] for i in filtered_indices]
        if printFlag : click.secho(f"Filtered {class_name} bboxes using threshold: {len(boxes)} --> {len(filtered_boxes)}", fg="blue")
        # Apply NMS on the filtered boxes
        selected_indices = apply_nms(filtered_boxes, scores, nms_threshold)
        # Get the final selected boxes after NMS
        nms_boxes = [filtered_boxes[i] for i in selected_indices]
        if printFlag : click.secho(f"Filtered {class_name} bboxes using NMS: {len(filtered_boxes)} --> {len(nms_boxes)}", fg="blue")
    else:
        nms_boxes = []
    return nms_boxes


if __name__ == "__main__":
    
    printFlag = True
    head_nms_boxes = []
    human_boxes = [
        [10, 20, 100, 200],  # Box 1
        [15, 25, 105, 205],  # Box 2 (overlapping with Box 1)
        [50, 60, 150, 250],  # Box 3
        [200, 220, 300, 350] # Box 4 (no overlap)
    ]
    human_scores = [0.9, 0.85, 0.95, 0.6]
    threshold_human = 0.8
    nms_threshold_human = 0.5

    # Example Usage
    human_nms_boxes = process_class_annotations('human', human_boxes, human_scores, threshold_human, nms_threshold_human, printFlag)

    # head_nms_boxes = process_class_annotations('head', head_boxes, head_scores, threshold_head, nms_threshold_head)

    # Combine the final bounding boxes for both classes
    final_boxes = human_nms_boxes + head_nms_boxes

    print(final_boxes)
