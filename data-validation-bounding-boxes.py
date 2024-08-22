import argparse, os, cv2, xml.etree.ElementTree as ET, json, click, numpy as np
from pathlib import Path
from tqdm import tqdm

''' This module loads the images and annotations and draw the bounding boxes on the images.'''

def parse_args():
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(description="Draw bounding boxes on images based on annotations.")
    parser.add_argument("--image-dir", type=str, required=True, help="Path to the directory containing the images.")
    parser.add_argument("--annotation-dir", type=str, required=True, help="Path to the directory containing the annotations.")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to the directory where output images will be saved.")
    parser.add_argument("--bbox-color", type=str, default="red", help="Color of the bounding boxes (e.g., 'red', 'green', 'blue').")
    parser.add_argument("--bbox-thickness", type=int, default=2, help="Thickness of the bounding boxes.")
    parser.add_argument("--text-color", type=str, default="white", help="Color of the text for labels (e.g., 'white', 'black').")
    parser.add_argument("--font-size", type=int, default=1, help="Font size for the labels.")
    return parser.parse_args()

def draw_bounding_boxes_with_legend(image_path, annotations, output_path, bbox_color, bbox_thickness, text_color, font_size):
    """Draw bounding boxes on the given image based on annotations and create a legend showing the colors used."""
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return
    # Define color map for bounding box and text colors
    color_map = {
        "red": (0, 0, 255),
        "green": (0, 255, 0),
        "blue": (255, 0, 0),
        "white": (255, 255, 255),
        "black": (0, 0, 0)
    }
    # Convert colors from strings to BGR tuples
    bbox_color = color_map.get(bbox_color.lower(), (0, 0, 255))
    text_color = color_map.get(text_color.lower(), (255, 255, 255))
    # Initialize a dictionary to map categories to colors
    category_colors = {}
    used_colors = set()
    color_index = 0
    # List of colors for categories
    colors_list = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (0, 255, 255),  # Cyan
        (255, 0, 255),  # Magenta
        (128, 128, 0),  # Olive
        (128, 0, 128),  # Purple
        (0, 128, 128)  # Teal
        # Add more colors as needed
    ]
    # Assign colors to categories and draw bounding boxes
    for annotation in annotations:
        xmin, ymin, xmax, ymax = annotation["bbox"]
        category = annotation["category"]

        # Assign a color to the category if not already assigned
        if category not in category_colors:
            # Use a color from the list and move to the next one
            if color_index >= len(colors_list):
                color_index = 0  # Loop back if we've used all colors
            category_colors[category] = colors_list[color_index]
            color_index += 1
        # Get the color for the current category
        color = category_colors[category]
        # Draw the bounding box
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, bbox_thickness)
    # Draw the legend
    x_legend = image.shape[1] - 150  # Adjust the x position of the legend
    y_legend = 10  # Initial y position for the legend
    line_height = 20  # Adjust the height of each line in the legend
    for category, color in category_colors.items():
        # Draw a square with the category color
        cv2.rectangle(image, (x_legend, y_legend), (x_legend + 15, y_legend + 15), color, -1)  
        # Draw the label text next to the square
        cv2.putText(image, category, (x_legend + 20, y_legend + 12), cv2.FONT_HERSHEY_SIMPLEX, font_size * 0.5, text_color, 1)
        # Move down to the next line in the legend
        y_legend += line_height
    # Save the image with bounding boxes and legend
    cv2.imwrite(output_path, image)
    # print(f"Saved output image: {output_path}")
    
def draw_bounding_boxes_with_top_text(image_path, annotations, output_path, bbox_color, bbox_thickness, text_color, font_size):
    """Draw bounding boxes on the given image based on annotations."""
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return
    
    # Define color
    color_map = {
        "red": (0, 0, 255),
        "green": (0, 255, 0),
        "blue": (255, 0, 0),
        "white": (255, 255, 255),
        "black": (0, 0, 0)
    }
    
    bbox_color = color_map.get(bbox_color.lower(), (0, 0, 255))
    text_color = color_map.get(text_color.lower(), (255, 255, 255))
    # Draw bounding boxes
    for annotation in annotations:
        xmin, ymin, xmax, ymax = annotation["bbox"]
        category = annotation["category"]
        # Draw rectangle (bounding box)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), bbox_color, bbox_thickness)
        # Draw text (label)
        cv2.putText(image, category, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, font_size, text_color, bbox_thickness)
    # Save the image with bounding boxes
    cv2.imwrite(output_path, image)
    # print(f"Saved output image: {output_path}")

def parse_xml_annotations(xml_file):
    """Parse annotations from an XML file."""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    annotations = []
    for obj in root.findall('object'):
        category = obj.find('name').text
        bndbox = obj.find('bndbox')    
        # Convert bounding box coordinates to floats first, then to integers
        xmin = int(float(bndbox.find('xmin').text))
        ymin = int(float(bndbox.find('ymin').text))
        xmax = int(float(bndbox.find('xmax').text))
        ymax = int(float(bndbox.find('ymax').text))
        annotation = {
            "category": category,
            "bbox": [xmin, ymin, xmax, ymax]
        }
        annotations.append(annotation)
    return annotations

def parse_yolo_annotations(yolo_file, image_shape):
    """Parse annotations from a YOLO format file."""
    annotations = []
    with open(yolo_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            category_id = int(parts[0])
            bbox_center_x = float(parts[1])
            bbox_center_y = float(parts[2])
            bbox_width = float(parts[3])
            bbox_height = float(parts[4])
            
            # Convert to absolute coordinates
            image_width, image_height = image_shape
            xmin = int((bbox_center_x - bbox_width / 2) * image_width)
            ymin = int((bbox_center_y - bbox_height / 2) * image_height)
            xmax = int((bbox_center_x + bbox_width / 2) * image_width)
            ymax = int((bbox_center_y + bbox_height / 2) * image_height)
            
            annotation = {
                "category": category_id,  # Adjust to your needs, e.g., category mapping
                "bbox": [xmin, ymin, xmax, ymax]
            }
            annotations.append(annotation)
    return annotations

def process_images(args):
    """Process images and draw bounding boxes."""
    image_dir = args.image_dir
    annotation_dir = args.annotation_dir
    output_dir = args.output_dir
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    # Get the list of image files
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    for image_file in tqdm(image_files):
        # Construct the image path
        image_path = os.path.join(image_dir, image_file)
        
        # Construct the annotation path based on the image filename
        xml_filename = os.path.splitext(image_file)[0] + ".xml"
        yolo_filename = os.path.splitext(image_file)[0] + ".txt"
        xml_path = os.path.join(annotation_dir, xml_filename)
        yolo_path = os.path.join(annotation_dir, yolo_filename)
        
        annotations = []
        if os.path.exists(xml_path):
            # Parse XML annotations
            annotations = parse_xml_annotations(xml_path)
        elif os.path.exists(yolo_path):
            # Parse YOLO annotations
            # Get image dimensions for YOLO format parsing
            image = cv2.imread(image_path)
            image_shape = image.shape[1], image.shape[0]
            annotations = parse_yolo_annotations(yolo_path, image_shape)
        else:
            print(f"No annotations found for image: {image_path}")
            continue
        # Construct the output image path
        output_path = os.path.join(output_dir, image_file)
        # Draw bounding boxes on the image
        draw_bounding_boxes_with_legend(image_path, annotations, output_path, args.bbox_color, args.bbox_thickness, args.text_color, args.font_size)

if __name__ == "__main__":
    args = parse_args()
    process_images(args)
    

# Execute code
'''
#### path setup
    DATA_PATH = f"C:/data/"

    # input
    num_images = 1000  # Number of random images to select
    input_image_folder = f"{DATA_PATH}images/"
    input_xml_folder = f"{DATA_PATH}annotations/"
    output_folder = f"{DATA_PATH}Transformed_Data_{num_images}/"


python data-validation-bounding-boxes.py --image-dir "C:/images/" --annotation-dir "C:/labels/" --output-dir "C:/annotations_validation/"

'''
