import os, sys, click, shutil, cv2
from tqdm import tqdm
import xml.etree.ElementTree as ET

''' This module loads the annotation files and eliminate the files not having the required classes.'''

def parse_xml_annotations(xml_file, required_classes):
    try:
        """Parse annotations from an XML file."""
        junkXmlFlag = True
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for obj in root.findall('object'):
            category = obj.find('name').text
            if category in required_classes:
                junkXmlFlag = False
                break
        print("---------------------------", junkXmlFlag)
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        click.secho(f"Exception : {e}\nError Type : {exc_type}\nFile Name : {fname}\nLine Number : {exc_tb.tb_lineno}", fg="red")
    return junkXmlFlag


def move_annotation_not_having_required_classes(images_dir, xml_dir, output_dir, required_classes):
    """
    Cuts images from the input directory and pastes them into the output directory.
    Args:
        image_file_names (list): List containing the names of image files to move.
        input_file_dir (str): Path to the input directory containing image files.
        output_dir (str): Path to the output directory where image files will be pasted.
    Returns:
        None: Moves the specified image files from the input directory to the output directory.
    """
    try:
        # Create the output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(f"{output_dir}images/"):
            os.makedirs(f"{output_dir}images")
        if not os.path.exists(f"{output_dir}labels/"):
            os.makedirs(f"{output_dir}labels")

        # Iterate through each image file name in the list
        click.secho(f"\n\nMoving files to the required directory : {output_dir}", fg="green")
        junkCounter = 0
        filesMovedCounter = 0
        for xml_file_name in tqdm(os.listdir(xml_dir)):
            # Construct the full path of the image file in the input directory
            image_file_name = xml_file_name.split(".")[0] + ".jpg"
            xml_file_path = os.path.join(xml_dir, xml_file_name)
            image_file_path = os.path.join(images_dir, image_file_name)

            junkXmlFlag = parse_xml_annotations(xml_file_path,
                                                required_classes)
            
            if junkXmlFlag:
                output_image_path = os.path.join(f"{output_dir}", "images/")
                output_xml_path = os.path.join(f"{output_dir}", "labels/")  
                junkCounter+=1
                
                # Check if the image file exists in the input directory
                # print(f"\n{xml_file_path} : \n{os.path.exists(xml_file_path)}")
                # print(f"\n{image_file_path} : \n{os.path.exists(image_file_path)}")

                try:    
                    shutil.move(xml_file_path, output_xml_path)
                    shutil.move(image_file_path, output_image_path)
                    filesMovedCounter+=1
                except Exception as e:
                    # Print a message if the file does not exist
                    click.secho(f"{image_file_path} or {xml_file_path} not found to move.", fg="red")

        # Print the count of missing annotation files
        print(f"Count of moved annotation files: {filesMovedCounter}")
        print("Annotation files not having required classes :", junkCounter)

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        click.secho(f"Exception : {e}\nError Type : {exc_type}\nFile Name : {fname}\nLine Number : {exc_tb.tb_lineno}", fg="red")


if __name__ == '__main__':
    
    # DATA_PATH = f""

    # input
    # num_images = 100  # Number of random images to select
    # input_image_folder = f"{DATA_PATH}images/"
    # input_xml_folder = f"{DATA_PATH}annotations/"
    # output_folder = f"{DATA_PATH}Transformed_Data_{num_images}/"


    # Define the source directory
    # DATA_PATH = ''
    
    dirname = f""
    DATA_PATH = f""
    xml_dir = f"{DATA_PATH}/annotations/"
    images_dir = f"{DATA_PATH}Jimages/"
    # directory to save images & labels if XML does not contain the required annotation classes
    output_dir = f"{DATA_PATH}Annotations-not-having-head-human-classes/"
    # required classes in the dataset
    required_classes = ["class1", "class2"]
    try:
        move_annotation_not_having_required_classes(images_dir,
                                                    xml_dir,
                                                    output_dir,
                                                    required_classes)
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        click.secho(f"Exception : {e}\nError Type : {exc_type}\nFile Name : {fname}\nLine Number : {exc_tb.tb_lineno}", fg="red")
