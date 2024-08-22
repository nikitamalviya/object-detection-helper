import os, sys, click, shutil, hashlib, cv2, matplotlib.pyplot as plt
from tqdm import tqdm

''' This module loads the images and annotations and draw the bounding boxes on the images to validate the annotations.'''

def collect_files_and_copy(source_dir, target_dir, formats):
    """
    Collects files with specified formats from a source directory (including subdirectories)
    and copies them to a target directory.
    Args:
        source_dir (str): Path to the source directory containing files in subdirectories.
        target_dir (str): Path to the target directory where files will be copied.
        formats (list or tuple): List of file extensions to look for, e.g., ['.jpg', '.jpeg', '.png'].
    Returns:
        None
    """
    # Create target directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # List to store paths of all files found
    file_paths = []

    # Walk through the source directory to find files
    for root, _, files in os.walk(source_dir):
        for file in files:
            # Check if the file has one of the supported formats
            if any(file.lower().endswith(extension) for extension in formats):
                # Construct the full path of the file
                file_path = os.path.join(root, file)
                
                # Add the file path to the list
                file_paths.append(file_path)

    # Copy files to the target directory
    click.secho(f"\n\nCopying {formats} files to the required directory : {target_dir}", fg="green")
    for file_path in tqdm(file_paths):
        # Extract the file name
        file_filename = os.path.basename(file_path)
        
        # Construct the destination path
        destination_path = os.path.join(target_dir, file_filename)
        
        # Copy the file to the target directory
        shutil.copy(file_path, destination_path)
        
        # Print a message indicating the file has been copied
        # print(f"Copied: {file_path} -> {destination_path}")


def cut_paste_images(image_file_names, input_file_dir, output_dir):
    """
    Cuts images from the input directory and pastes them into the output directory.
    Args:
        image_file_names (list): List containing the names of image files to move.
        input_file_dir (str): Path to the input directory containing image files.
        output_dir (str): Path to the output directory where image files will be pasted.
    Returns:
        None: Moves the specified image files from the input directory to the output directory.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate through each image file name in the list
    click.secho(f"\n\nMoving files to the required directory : {output_dir}", fg="green")
    for image_file_name in tqdm(image_file_names):
        # Construct the full path of the image file in the input directory
        input_file_path = os.path.join(input_file_dir, image_file_name)
        
        # Construct the destination path in the output directory
        output_file_path = os.path.join(output_dir, image_file_name)
        
        # Check if the image file exists in the input directory
        if os.path.exists(input_file_path):
            # Move the image file from the input directory to the output directory
            shutil.move(input_file_path, output_file_path)
            # print(f"Moved {image_file_name} from {input_file_dir} to {output_dir}")
        else:
            # Print a message if the file does not exist
            print(f"{image_file_name} not found in {input_file_dir}")


def find_duplicates(folder_name):
    '''
    This function finds the duplicates in an image folder.
    The folder will contain 'n' number of images, from which duplicate images will be removed.
    @param folder_name : name of the folder containing images
    returns (duplicates_name, duplicates)
    '''
    try:
        if not os.path.exists(folder_name):
            print(f"Image folder {folder_name} does not exists !")
            return (0, 0, 0)
        # extract all images 
        all_images = os.listdir(folder_name)
        # to store duplicates
        duplicates = []
        # to store duplicate names
        duplicates_name = []
        # to store hash values
        hash_keys = dict()

        click.secho(f"\n\nFinding duplicates in directory : {folder_name}", fg="green")
        # find duplicates inside the directory
        for index, filename in  tqdm(enumerate(os.listdir('.'))):  #listdir('.') = current directory
            if os.path.isfile(filename):
                with open(filename, 'rb') as f:
                    # convert in image hash value
                    filehash = hashlib.md5(f.read()).hexdigest()
                # add hash key of an image if not present
                if filehash not in hash_keys:
                    hash_keys[filehash] = index
                # if already present, then consider it to be duplicate
                else:
                    duplicates.append((index, hash_keys[filehash]))
                    duplicates_name.append(filename)
        print(f"\nTotal images : {len(all_images)}, Duplicates found : {len(duplicates)}\nDuplicates list : {duplicates_name}")
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        click.secho(f"Exception : {e}\nError Type : {exc_type}\nFile Name : {fname}\nLine Number : {exc_tb.tb_lineno}", fg="red")
    return duplicates_name, duplicates, all_images


def check_missing_annotations(image_dir, annotation_dir, annotation_format):
    """
    Checks for missing annotation files for images and prints the count and names of missing files.
    Args:
        image_dir (str): Path to the directory containing image files.
        annotation_dir (str): Path to the directory containing annotation files.
    Returns:
        None: Prints the count and names of missing annotation files.
    """
    try:
        # Initialize count of missing annotation files
        annotation_count = 0
        # List to hold the names of missing annotation files
        missing_files = []
        # Get the list of image files in the image directory
        image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

        print(f"Number of Images found : {len(image_files)}")       
        # Iterate through each image file
        click.secho(f"\n\nChecking images and corresponding annotations files...", fg="green")
        for image_file in tqdm(image_files):
            # Extract the base name (without extension) of the image file
            image_base_name = os.path.splitext(image_file)[0]
            # annotation format
            annotation_file_name = f"{image_base_name}.{annotation_format}"  
            # Construct the path to the annotation file
            annotation_file_path = os.path.join(annotation_dir, annotation_file_name)
            
            # Check if the annotation file exists
            if not os.path.exists(annotation_file_path):
                # Add the missing file name to the list
                missing_files.append(image_file)
            else:
                annotation_count+=1

        # Print the count of missing annotation files
        print(f"Total image files : {len(image_files)}")
        print(f"Total XML files : {annotation_count}")
        print(f"Count of missing annotation files: {len(missing_files)}")
        print("Missing annotation files:", missing_files)
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        click.secho(f"Exception : {e}\nError Type : {exc_type}\nFile Name : {fname}\nLine Number : {exc_tb.tb_lineno}", fg="red")
    return missing_files


if __name__ == '__main__':
    
    ''' Movement of data from multiple folders to image and label folder '''
    movement_required = False

    # Define the source directory containing images in sub-directories
    source_dir = 'C:/images/'
    
    # Define the target directory where images will be copied
    annotations_dir = "C:/annotations/"
    # image dir path
    images_dir = source_dir
    # dir to save images which do not annotations present
    no_annotations_dir = "C:/imagesWithNoAnnotations/"

    try:
        if not os.path.exists(no_annotations_dir):
            os.makedirs(no_annotations_dir)
            
        # Collect images and XML labels to copy them to the target directory
        if movement_required:
            # collect images
            collect_files_and_copy(source_dir=source_dir,
                                    target_dir=images_dir,
                                    formats=('.jpg', '.jpeg', '.png'))
            # collect XMLS
            collect_files_and_copy(source_dir=source_dir,
                                    target_dir=annotations_dir,
                                    formats=('.xml'))
            
        # find duplicates images in the dataset if any
        duplicates_name, duplicates, all_images = find_duplicates(folder_name=images_dir)
        
        # Call the function to check for missing annotations
        missing_files = check_missing_annotations(image_dir=images_dir,
                                                  annotation_dir=annotations_dir,
                                                  annotation_format="xml")

        # Cut paste or move the files to another dir whose annotations are not present
        cut_paste_images(image_file_names=missing_files,
                        input_file_dir=images_dir,
                        output_dir=no_annotations_dir)

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        click.secho(f"Exception : {e}\nError Type : {exc_type}\nFile Name : {fname}\nLine Number : {exc_tb.tb_lineno}", fg="red")
