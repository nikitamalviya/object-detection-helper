''' This module visualizes the bounding boxes on an image using the annotations having bounding boxes, confidence score and category.
It populates the drawn bounding boxes as legend on the top left side of the image.
It displays the bounding boxes with different colors for different categories. '''

import os, sys, click, io, cv2, numpy as np

def draw_bounding_boxes_with_labels_on_top(image, annotations, class_colors, output_path):
    try:
        # Initialize text for labels
        label_text = ""
        for i, annotation in enumerate(annotations, 1):
            # Extract bounding box coordinates
            bbox = annotation['bbox']
            x1, y1, x2, y2 = bbox
            # Extract category and score
            category = annotation['category']
            score = annotation['score']
            # Get the color for the current category
            color = class_colors.get(category, (255, 255, 255))  # Default to white if category not found
            # Draw the bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            # Put the box number on the bounding box
            cv2.putText(image, str(i), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            # Append to label text
            label_text += f"{i}: {category} ({score:.2f}), "
        # Remove trailing comma and space
        label_text = label_text.rstrip(', ')
        # Put all labels at the top of the image
        y0, dy = 30, 30
        for i, line in enumerate(label_text.split(', ')):
            y = y0 + i*dy
            cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        # Save the image with bounding boxes to the same path
        cv2.imwrite(output_path, image)
        click.secho(f"Output image saved at local path : {image_path}", fg="green")
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        click.secho(f"Unable to draw bounding box.\nException: {e}\nError Type: {exc_type}\nFile Name: visualization_ai_vision_utility.py\nLine Number: {exc_tb.tb_lineno}", fg="red")
    return image


if __name__ == "__main__":

    # A dictionary with class names as keys and colors as values
    class_colors = {
    "human": (0, 255, 0),    # Green
    "head": (0, 0, 255),     # Red
    "car": (255, 0, 0),      # Blue (OpenCV uses BGR, so blue is (255, 0, 0) in RGB)
    "dog": (0, 255, 255),    # Yellow (OpenCV uses BGR, so yellow is (0, 255, 255) in RGB)
    "cat": (255, 0, 255),    # Magenta
    "bike": (255, 255, 0),   # Cyan (OpenCV uses BGR, so cyan is (255, 255, 0) in RGB)
    "bus": (255, 255, 255), # White
    "truck": (0, 165, 255),  # Orange (OpenCV uses BGR, so orange is (0, 165, 255) in RGB)
    "traffic light": (128, 128, 0), # Teal
    "fire hydrant": (42, 42, 165), # Dark Brown
    "stop sign": (180, 105, 255) # Hot Pink
    # (100, 149, 237) # Cobalt Blue
    # (0, 128, 0)  # Dark Green
    # (0, 128, 128) # Olive
    # (128, 0, 128) # Purple
    # (54, 69, 79), # Charcoal
    }
    

    # Load the image
    PATH = f"./data/"
    image_path = f"{PATH}sample_test_image.jpg"
    output_path = f"{PATH}visualized_image.jpg"

    image = cv2.imread(image_path)
    print(image.shape)

    annotations = [
    {'id': '1', 'bbox': [50, 50, 200, 400], 'category': 'human', 'score': 0.93},
    {'id': '2', 'bbox': [220, 60, 370, 420], 'category': 'head', 'score': 0.91},
    {'id': '3', 'bbox': [600, 200, 640, 300], 'category': 'car', 'score': 0.92},
    {'id': '4', 'bbox': [450, 500, 700, 800], 'category': 'car', 'score': 0.90},
    {'id': '5', 'bbox': [800, 50, 950, 250], 'category': 'dog', 'score': 0.85},
    {'id': '6', 'bbox': [100, 700, 250, 850], 'category': 'cat', 'score': 0.88},
    {'id': '7', 'bbox': [700, 300, 780, 500], 'category': 'bike', 'score': 0.87},
    {'id': '8', 'bbox': [200, 500, 400, 800], 'category': 'bus', 'score': 0.89},
    {'id': '9', 'bbox': [1000, 300, 1100, 700], 'category': 'truck', 'score': 0.91},
    {'id': '10', 'bbox': [50, 600, 150, 900], 'category': 'traffic light', 'score': 0.86},
    {'id': '11', 'bbox': [600, 50, 700, 200], 'category': 'fire hydrant', 'score': 0.83},
    {'id': '12', 'bbox': [850, 600, 1000, 900], 'category': 'stop sign', 'score': 0.84},
    ]


    pred_bbox_image = draw_bounding_boxes_with_labels_on_top(image,
                                                             annotations,
                                                             class_colors,
                                                             output_path)
