import os, sys, click, random, shutil, argparse
from tqdm import tqdm
from pathlib import Path

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Divide and split dataset into train, test, and valid sets.")
    parser.add_argument("--image-dir", type=str, required=True, help="Path to the directory containing images.")
    parser.add_argument("--label-dir", type=str, required=True, help="Path to the directory containing labels.")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to the output directory where split datasets will be saved.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Ratio of the dataset to use for training (default: 0.8).")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Ratio of the dataset to use for testing (default: 0.2).")
    parser.add_argument("--valid-ratio", type=float, default=0.0, help="Ratio of the dataset to use for validation (default: 0.0).")
    parser.add_argument("--random", type=bool, default=True, help="If set True, the data will be randomized before splitting.")
    return parser.parse_args()

def calculate_image_splits(num_images, train_ratio, test_ratio, valid_ratio):
    # Ensure the ratios sum to 1
    assert train_ratio + test_ratio + valid_ratio == 1, "The sum of the ratios must be 1."
    
    # Calculate the sizes
    train_size = int(num_images * train_ratio)
    test_size = int(num_images * test_ratio)
    valid_size = int(num_images * valid_ratio)
    
    # Adjust for any rounding errors to ensure the total equals num_images
    total_size = train_size + test_size + valid_size
    if total_size < num_images:
        train_size += num_images - total_size
    elif total_size > num_images:
        train_size -= total_size - num_images
    return train_size, test_size, valid_size
    
def calculate_image_splits_incorrect(num_images, train_ratio, test_ratio, valid_ratio):
    """
    Calculate the number of images for train, test, and valid sets based on specified ratios.
    Args:
        num_images (int): Total number of images.
        train_ratio (float): Ratio for the training set.
        test_ratio (float): Ratio for the testing set.
        valid_ratio (float): Ratio for the validation set.
    Returns:
        tuple: A tuple containing the number of images in the training set, testing set, and validation set respectively.
    """
    # Ensure ratios sum to 1
    if train_ratio + test_ratio + valid_ratio == 1:
        raise ValueError("Ratios must sum to 1. Please provide valid ratios.")

    # Calculate total ratio for train and test sets (excluding validation)
    total_split_ratio = train_ratio + test_ratio + valid_ratio

    # Calculate number of images for each split
    train_size = int(num_images * train_ratio / total_split_ratio)
    test_size = int(num_images * test_ratio / total_split_ratio)
    valid_size = int(num_images * valid_ratio)
    
    print(f"--- Train size: {train_size}, Test size: {test_size}, Valid size: {valid_size}")
    return train_size, test_size, valid_size


def split_dataset(image_dir, label_dir, output_dir, train_ratio, test_ratio, valid_ratio, randomize=False):
    """
    Splits and moves the dataset of images and corresponding labels into training, testing, and validation sets based on specified ratios.
    Args:
        image_dir (str): Path to the directory containing the image files.
        label_dir (str): Path to the directory containing the label files (e.g., annotations).
        output_dir (str): Path to the output directory where the split data will be saved. Subdirectories for train, test, and valid sets will be created within this directory.
        train_ratio (float): Ratio of the dataset to be used for training (between 0 and 1).
        test_ratio (float): Ratio of the dataset to be used for testing (between 0 and 1).
        valid_ratio (float): Ratio of the dataset to be used for validation (between 0 and 1). Must sum to 1 with train_ratio and test_ratio.
        randomize (bool, optional): If True, shuffles the data before splitting. Defaults to False.
    Raises:
        AssertionError: If the number of images and labels do not match.
        ValueError: If the provided ratios do not sum to 1.
        Exception: Any other exceptions encountered during the process.
    """
    # Create output directories for train, test, and valid sets
    train_output_dir = Path(output_dir) / 'train'
    test_output_dir = Path(output_dir) / 'test'
    valid_output_dir = Path(output_dir) / 'valid'
    
    # Create the main directories if they don't exist
    train_output_dir.mkdir(parents=True, exist_ok=True)
    test_output_dir.mkdir(parents=True, exist_ok=True)
    valid_output_dir.mkdir(parents=True, exist_ok=True)

    # Create images and labels directories inside each of the main directories
    (train_output_dir / 'images').mkdir(parents=True, exist_ok=True)
    (train_output_dir / 'labels').mkdir(parents=True, exist_ok=True)
    (test_output_dir / 'images').mkdir(parents=True, exist_ok=True)
    (test_output_dir / 'labels').mkdir(parents=True, exist_ok=True)
    (valid_output_dir / 'images').mkdir(parents=True, exist_ok=True)
    (valid_output_dir / 'labels').mkdir(parents=True, exist_ok=True)

    # List of images and labels
    images = []
    labels = []

    # Populate images and labels lists
    for image_file in os.listdir(image_dir):
        if image_file.endswith(('.jpg', '.jpeg', '.png')):
            # Check if there is a corresponding annotation file
            annotation_file = os.path.splitext(image_file)[0] + '.xml'
            annotation_path = os.path.join(label_dir, annotation_file)
            if os.path.exists(annotation_path):
                images.append(image_file)
                labels.append(annotation_file)
    
    # Ensure both lists have the same number of elements
    assert len(images) == len(labels), "Number of images and labels do not match."

    # Combine images and labels into a list of tuples
    data = list(zip(images, labels))

    # Randomize the data if required
    if randomize:
        random.shuffle(data)

    # Calculate the number of images for each set
    num_images = len(images)
    train_size, test_size, valid_size = calculate_image_splits(num_images,
                                                               train_ratio,
                                                               test_ratio,
                                                               valid_ratio)
    # Print out the sizes for debugging
    print(f"Train size: {train_size}, Test size: {test_size}, Valid size: {valid_size}")

    # Correctly slice the data into train, test, and valid sets
    train_data = data[:train_size]
    test_data = data[train_size:train_size + test_size]
    valid_data = data[train_size + test_size:]

    # Print out the length of each dataset for debugging
    print(f"Data split into: Train data length: {len(train_data)}, Test data length: {len(test_data)}, Valid data length: {len(valid_data)}")

    # Function to move files to the specified output directory
    def move_files(data, image_output_dir, label_output_dir):
        for image_file, label_file in tqdm(data):
            # Move image
            shutil.copy(os.path.join(image_dir, image_file), image_output_dir)
            # Move label
            shutil.copy(os.path.join(label_dir, label_file), label_output_dir)

    try:
        # Move data to the corresponding directories
        click.secho(f"\nPreparing training data...", fg="green")
        move_files(train_data, train_output_dir / 'images', train_output_dir / 'labels')
        click.secho(f"\nPreparing testing data...", fg="green")
        move_files(test_data, test_output_dir / 'images', test_output_dir / 'labels')
        click.secho(f"\nPreparing validation data...", fg="green")
        move_files(valid_data, valid_output_dir / 'images', valid_output_dir / 'labels')
        print(f"Dataset split completed: {len(train_data)} train, {len(test_data)} test, {len(valid_data)} valid.")

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        click.secho(f"Exception: {e}\nError Type: {exc_type}\nFile Name: {fname}\nLine Number: {exc_tb.tb_lineno}", fg="red")


def split_dataset_train_test_val(image_dir, label_dir, output_dir, train_value, test_value, val_value, randomize=False, mode="ratio"):
    # Ensure the mode is valid
    if mode not in ["ratio", "count"]:
        raise ValueError("Mode must be either 'ratio' or 'count'")

    # Get the list of images and labels
    images = sorted([f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))])
    labels = sorted([f for f in os.listdir(label_dir) if os.path.isfile(os.path.join(label_dir, f))])
    
    if randomize:
        combined = list(zip(images, labels))
        random.shuffle(combined)
        images, labels = zip(*combined)
    
    total_images = len(images)
    total_labels = len(labels)
    
    # Check if the number of images and labels are equal
    if total_images != total_labels:
        raise ValueError("The number of images and labels must be the same")

    if mode == "ratio":
        train_count = int(total_images * train_value)
        test_count = int(total_images * test_value)
        valid_count = int(total_images * val_value)
    elif mode == "count":
        train_count = train_value
        test_count = test_value
        valid_count = val_value
    
    click.secho(f"Split count..........", fg="green")
    print(train_count, test_count, valid_count, total_images, train_count + test_count + valid_count)

    # Adjust counts to ensure they add up to the total number of images
    diff = total_images - (train_count + test_count + valid_count)
    click.secho(f"Difference in total and split count..........", fg="green")
    print(diff, train_count + test_count + valid_count)
    
    if diff > 0:
        train_count += diff
    elif diff < 0:
        raise ValueError("The sum of train, test, and validation counts/ratios exceeds the total number of images")
    

    # Create output directories if they don't exist
    for split in ["train", "test", "val"]:
        os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, "labels"), exist_ok=True)
    
    # Helper function to copy files
    def copy_files(start_idx, end_idx, split):
        for i in range(start_idx, end_idx):
            shutil.copy(os.path.join(image_dir, images[i]), os.path.join(output_dir, split, "images", images[i]))
            shutil.copy(os.path.join(label_dir, labels[i]), os.path.join(output_dir, split, "labels", labels[i]))
    

    # Split the data
    copy_files(0, train_count, "train")
    copy_files(train_count, train_count + test_count, "test")
    if valid_count > 0:
        copy_files(train_count + test_count, train_count + test_count + valid_count, "val")


def main():
    # Parse the command-line arguments
    args = parse_args()

    # Split the dataset into train, test, and valid sets based on specified ratios
    split_dataset(args.image_dir, args.label_dir, args.output_dir, args.train_ratio, args.test_ratio, args.valid_ratio, args.random)


if __name__ == "__main__":
    
     # Define the source directory
    PATH = 'C:/Users/nvl3kor/OneDrive - Bosch Group/Work/Project/Kubota/data/'

    label_dir = f"{PATH}annotations/"
    images_dir = f"{PATH}Jimages/"
    output_dir = f"{PATH}Splitted Data/Cycle_2/"

    # train_value = 0.6
    # test_value = 0.4
    # val_value = 0.0

    mode="count" #'ratio'
    train_value = 6762 # 9262
    test_value = 2500 # 0
    val_value = 0

    split_dataset_train_test_val(images_dir,
                                 label_dir,
                                 output_dir,
                                 train_value,
                                 test_value,
                                 val_value,
                                 randomize=True,
                                 mode=mode)

    # main()



'''
python data-split-into-train-test-val.py --image-dir "C:/images/" --label-dir "C:/annotations/" --output-dir "C:/Prepared Data/" --train-ratio 27 --test-ratio 73 --valid-ratio 0 --random True


Test: 73.00%  --> 6762 images
Train: 27%  --> 2500 images
'''
