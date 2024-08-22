import os, json, click, sys, xml.etree.ElementTree as ET
from collections import defaultdict
from pprint import pprint 
import matplotlib.pyplot as plt

def get_xml_updated_class_count(xml_file, unique_classes, classes_count):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for obj in root.findall('object'):
        category_name = obj.find('name').text
        if category_name not in unique_classes:
            unique_classes.append(category_name)
            classes_count[category_name]=0
        else:
            classes_count[category_name]+=1
    return unique_classes, classes_count

def get_bar_plot_for_class_value_counts(data):
    # Extract keys and values from the dictionary
    keys = list(data.keys())
    values = list(data.values())

    # Create a bar plot
    plt.bar(keys, values, color='steelblue')

    # Annotate each bar with its value
    for index, value in enumerate(values):
        plt.text(index, value // 2, str(value), ha='center', va='bottom', color='white', fontsize=8)

    # Add title and labels
    plt.title('Classes value counts')
    plt.xlabel('Classes')
    plt.ylabel('Value Counts')

    # Specify the file path where you want to save the plot
    file_path = './class_value_counts.png'  # You can specify a different file path and format (e.g., bar_plot.jpg)
    # Save the plot
    plt.savefig(file_path)
    # Show the plot
    plt.show()

def process_directory(xml_dir):
    try:
        # get class details
        unique_classes = []
        classes_count = {}
        for filename in os.listdir(xml_dir):
            # print(filename)
            if filename.endswith('.xml'):
                xml_path = os.path.join(xml_dir, filename)
                unique_classes, classes_count = get_xml_updated_class_count(xml_path,
                                                                            unique_classes,
                                                                            classes_count)
        
        get_bar_plot_for_class_value_counts(data=classes_count)
        # sort the classes value_counts                                                                            
        classes_count = sorted(classes_count.items(), key=lambda item: item[1], reverse=True)
        print(f"Unique classes : {unique_classes}")
        click.secho(f"Classes value_counts : ", fg="green")
        pprint(classes_count)

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        click.secho(f"\nError Type : {exc_type} ----->> File Name : {fname} ----->> Line Number : {exc_tb.tb_lineno}", fg="red")


if __name__ == "__main__":
    # annotations_dir = ""
    # process_directory(annotations_dir)

    click.secho(f"\nTraining Dataset Classes details : ", fg="green")
    annotations_dir = ""
    process_directory(annotations_dir)

    print("\n\n")

    click.secho(f"Testing Dataset Classes details : ", fg="green")
    annotations_dir = ""
    process_directory(annotations_dir)
