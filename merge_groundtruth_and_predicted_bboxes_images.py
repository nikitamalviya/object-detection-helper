''' This module merges the two images into one, for comparative analysis of ground truth and predicted bounding boxes. '''

from PIL import Image
import os, sys, cv2, numpy as np

def merge_gt_and_pred_bbox_images(cv2_image1, cv2_image2):
    # Convert the OpenCV images to PIL format
    image1 = Image.fromarray(cv2.cvtColor(cv2_image1, cv2.COLOR_BGR2RGB))
    image2 = Image.fromarray(cv2.cvtColor(cv2_image2, cv2.COLOR_BGR2RGB))
    # Determine the width and height of the final image
    total_width = image1.width + image2.width
    max_height = max(image1.height, image2.height)
    # Create a new image with a white background
    new_image = Image.new('RGB', (total_width, max_height), (255, 255, 255))
    # Paste image1 at (0, 0)
    new_image.paste(image1, (0, 0))    
    # Paste image2 next to image1
    new_image.paste(image2, (image1.width, 0))
    # Convert the merged PIL image back to OpenCV format
    merged_cv2_image = cv2.cvtColor(np.array(new_image), cv2.COLOR_RGB2BGR)
    return merged_cv2_image


def merge_images(image1_path, image2_path, output_path):
    # Open both images
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)
    # Determine the width and height of the final image
    total_width = image1.width + image2.width
    max_height = max(image1.height, image2.height)
    # Create a new image with a white background
    new_image = Image.new('RGB', (total_width, max_height), (255, 255, 255))
    # Paste image1 at (0, 0)
    new_image.paste(image1, (0, 0))
    # Paste image2 next to image1
    new_image.paste(image2, (image1.width, 0))
    # Save the merged image
    new_image.save(output_path)
    print(f'Merged image saved as {output_path}')

if __name__ == '__main__':
    # Paths to the images
    PATH = f"C:/Users/nvl3kor/OneDrive - Bosch Group/Work/Personal/data/"
    image1_path = f'{PATH}15_Camera1_236_mini_20230803172803_01542.jpg'
    image2_path = f'{PATH}15_Camera1_236_mini_20230803172803_02007.jpg'
    output_path = f'{PATH}merged_image.jpg'

    # Merge the images
    merge_images(image1_path, image2_path, output_path)
