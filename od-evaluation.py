# click.secho(f"\nUnable to ....", fg="red")
# dbutils.library.restartPython()
%pip install cognitive_service_vision_model_customization_python_samples
import os, sys, click, io, json, traceback, numpy as np, pandas as pd
from datetime import datetime
from PIL import Image
from tqdm import tqdm
from azure.storage.blob import BlobServiceClient

from prediction_ai_vision_utility_updated import (predict_image, parse_detection_results)
from visualization_ai_vision_utility import (draw_bounding_boxes_with_labels_on_top)
from compare_two_images import (merge_gt_and_pred_bbox_images)
from evaluation_utils import (parse_annotation,
                              match_predictions_to_ground_truth,
                              calculate_model_performance,
                              get_model_performance_dataframe,
                              compute_overall_model_performance,
                              save_overall_results_to_an_excel)
from blob_dbfs_file_utils import (ensure_directory_exists,
                                  check_file_exists, 
                                  read_file_from_dbfs,
                                  read_image_from_dbfs,
                                  read_xml_from_dbfs,
                                  save_file_to_dbfs,
                                  save_image_to_dbfs,
                                  save_csv_to_dbfs)

# AI Vision Service Configuration
resource_name = "abcd"
resource_key = ''
model_name = "model-v1"

# Storage Blob Configuration
storage_account_name = 'storage'
container_name = ''
account_key = ''
connection_string = f'DefaultEndpointsProtocol=https;AccountName=d1josajw01kbtcom;AccountKey={account_key};EndpointSuffix=core.windows.net'

# Create a BlobServiceClient object using the connection string
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_client = blob_service_client.get_container_client(container_name)

mount_dir = f"{container_name}"
try:
    dbutils.fs.unmount(f"/mnt/{mount_dir}")
except:
    pass
# Mount the storage
dbutils.fs.mount(
    source = f"wasbs://{mount_dir}@{storage_account_name}.blob.core.windows.net",
    mount_point = f"/mnt/{mount_dir}",  # Chosen mount point
    extra_configs = {f"fs.azure.account.key.{storage_account_name}.blob.core.windows.net": account_key}
)
print(display(dbutils.fs.ls(f"/mnt/{mount_dir}")))

##################### CONFIGURATION SETTINGS ######################

LIMIT_COUNT = 1 #"no_limit" 

#### Evaluation dataset
evaluation_set = f"Container_Name"

BLOB_PATH = f"/mnt/{mount_dir}/"
blobName = f"{BLOB_PATH}{evaluation_set}"
# Input Blob Paths
inputBlob = f'{blobName}/Path1/'
annotationBlob = f"{blobName}/Path2/"
if LIMIT_COUNT == "no_limit":
    LIMIT_COUNT = len([file.path for file in dbutils.fs.ls(inputBlob)])

#### Flags Configuration
printFlag = True #False
drawBbox = True # visualization
saveExcelFlag = True #True  # save results in excel

#### Prediction Configuration
required_classes = ["class1", "class2"]
threshold_head = 0.3
threshold_human = 0.5
# NMS threshold to filter the final prediction results
nms_threshold_human=0.65
nms_threshold_head=0.97

#### Evaluation Configuration
# iou threshold for matching the ground truth bboxes and prediction bboxes
human_iou_threshold = 0.5
# head confidence score for matching the ground truth bboxes and prediction bboxes
head_iou_threshold=0.01

#### Output Blob Path
mainOutputDir = f"{model_name}-THRESHHUMAN_{threshold_human}-THRESHHEAD_{threshold_head}-NMSHUMAN_{nms_threshold_human}_NMSHEAD_{nms_threshold_head}"
outputBlob = f"{BLOB_PATH}ImageAnalysisDataset/evaluation_results/{mainOutputDir}/MODEL_{model_name}-EVALSET_{evaluation_set}-COUNT_{LIMIT_COUNT}-THRESHHUMAN_{threshold_human}-THRESHHEAD_{threshold_head}-NMSHUMAN_{nms_threshold_human}_NMSHEAD_{nms_threshold_head}-DATE_{datetime.today().strftime('%d%m%Y')}/"
print("\noutputBlob : ", outputBlob)

#### Generated Outputs Blob Paths
# stores predicted bbox images in blob 
outputPredictionImgBlob = f"{outputBlob}OverlayGeneration/"
# stores predicted response in JSON in blob 
outputJsonBlob = f"{outputBlob}JSONs/"
# stores combined predicted bbox images & groundtruth bbox images in blob
outputEvaluationImgBlob = f"{outputBlob}Evaluation/"
# stores generated groundtruth results
xmlBboxAnnotations = f"{blobName}/Visualization/"

# Create the above mentioned directories if they doesn't exist
for dirname in [outputPredictionImgBlob, outputJsonBlob, outputEvaluationImgBlob, xmlBboxAnnotations]:
    ensure_directory_exists(dirname)

###### Evaluation Excel File Paths
outputImgwiseExcel = f"{outputBlob}eval_imagewise-MODEL{model_name}-EVALSET_{evaluation_set}-COUNT_{LIMIT_COUNT}-THRESHHUMAN_{threshold_human}-THRESHHEAD_{threshold_head}-NMSHUMAN_{nms_threshold_human}_NMSHEAD_{nms_threshold_head}-DATE_{datetime.today().strftime('%d%m%Y')}.csv"
outputOverallExcel = f"{outputBlob}eval_overall_{model_name}-EVALSET_{evaluation_set}-COUNT_{LIMIT_COUNT}-THRESHHUMAN_{threshold_human}-THRESHHEAD_{threshold_head}-NMSHUMAN_{nms_threshold_human}_NMSHEAD_{nms_threshold_head}-DATE_{datetime.today().strftime('%d%m%Y')}.csv"


''' Batch Evaluation '''
if __name__ == "__main__":

    # Class-Based Metrics Tracking
    class_metrics={}
    for label in required_classes:
        class_metrics[label] = {"precision": [], "recall": [], "average_iou": [], "tp": 0, "fp": 0, "fn": 0}

    #### Iterate over the images
    # List and load image paths
    print("\ninputBlob : ", inputBlob)
    # image_blobs = container_client.list_blobs(name_starts_with=inputBlob)
    # image_paths = [blob.name for blob in image_blobs]
    image_paths = [file.path for file in dbutils.fs.ls(inputBlob)]

    print(f"Number of loaded images from set {inputBlob} : {len(image_paths)}\n{'#'*30}")

    # save evaluation results
    imagewise_results = []
    # Loop over image paths to load the corresponding XML file
    # for iter_, image_path in enumerate(t tqdm(image_paths, desc="Processing Images")):
    for iter_, image_path in enumerate(tqdm(image_paths, desc="Status: ")):
        try:
            click.secho(f"*"*90, fg="yellow")            
            if iter_ == LIMIT_COUNT:
                break
            print(iter_, LIMIT_COUNT)

            # click.secho(f"\nimage_path : {image_path}", fg="green")
            # Provide the image path from the storage account container

            image_name = image_path.split('/')[-1]

            filename = image_name.split(".")[0]
            image_path = f'{inputBlob}{image_name}'
            click.secho(f"\nimage_path : {image_path}", fg="green")
            ####### outputs : stores generated groundtruth results #######
            groundtruthBboxPath = f"{xmlBboxAnnotations}{image_name}"
            # stores predicted bbox images in blob 
            predictionPath = f"{outputPredictionImgBlob}{image_name}"
            # stores predicted response in JSON in blob
            jsonPath = f"{outputJsonBlob}{filename}.json"
            # stores combined predicted bbox images & groundtruth bbox images in blob
            mergedBboxPath = f"{outputEvaluationImgBlob}{image_name}"

            ###### API Prediction ######
            # read image
            img_content = read_file_from_dbfs(image_path)

            ####### Load and Parse the XML file ######
            xml_path = f"{annotationBlob}{filename}.xml"
            xml_tree = read_xml_from_dbfs(xml_path)                     
            groundtruths_annotations = parse_annotation(xml_tree, required_classes=required_classes)
            # print(f"\n\ngroundtruths_annotations : {groundtruths_annotations}\n")

            ####### Visualize the bbox for GT bounding boxes ###### groundtruthBboxPath
            image = read_image_from_dbfs(image_path)
            gt_bbox_image = draw_bounding_boxes_with_labels_on_top(image, groundtruths_annotations)
            gtStatus = save_image_to_dbfs(gt_bbox_image, groundtruthBboxPath)
            # if (not check_file_exists(groundtruthBboxPath)): # if ground truth visualization does not exists 
            # else: click.secho(f"XML bounding box annotations exits!", fg="blue")


            ####### Model Prediction on the image #######
            api_response = predict_image(img_content,
                                         model_name,
                                         resource_name,
                                         resource_key)
            # print("\napi_response : ", api_response)

            ## Filter Predicted Bboxes Apply Thresholds and NMS threshold to the results
            # Parse the output
            azure_results, pred_results = parse_detection_results(api_response,
                                                                  model_name,
                                                                  threshold_human,
                                                                  threshold_head,
                                                                  nms_threshold_human,
                                                                  nms_threshold_head)
            # print("\nParsed JSON Response : \n", json.dumps(pred_results, indent=2))

            ####### Visualize the PREDICTED bboxes ######
            image = read_image_from_dbfs(image_path)
            pred_bbox_image = draw_bounding_boxes_with_labels_on_top(image, pred_results["annotations"])
            predStatus = save_image_to_dbfs(pred_bbox_image, predictionPath)
            
            ####### Save JSON for stor model prediction respone ######
            if isinstance(pred_results, dict): jsonContent = json.dumps(pred_results, indent=4)
            save_file_to_dbfs(jsonContent, jsonPath)
            
            ####### Merge the Prediction and GT image for analysis ######
            # Compare the bounding boxes of prediction & ground truth -- Merge the images
            try:
                if gtStatus and predStatus:
                    merged_cv2_image = merge_gt_and_pred_bbox_images(cv2_image1=read_image_from_dbfs(groundtruthBboxPath),
                                                                     cv2_image2=read_image_from_dbfs(predictionPath))
                    mergedStatus = save_image_to_dbfs(merged_cv2_image, mergedBboxPath)
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                line_number = exc_tb.tb_lineno
                click.secho(f"Unable to Merge the GT and Pred image!!\nException: {e} ----->> Error Type: {exc_type} ----->> File Name: {fname} ----->> Line Number: {line_number}", fg="red")

            ####### Append prediction results for excel generation ######
            image_result = {"File Name": image_name,
                            "GT Objects": len(groundtruths_annotations),
                            "Predicted Objects": len(pred_results["annotations"])}
            
            result_dict1, result_dict2 = match_predictions_to_ground_truth(groundtruths_annotations,
                                                                           pred_results["annotations"],
                                                                           required_classes,
                                                                           human_iou_threshold=human_iou_threshold, 
                                                                           human_pred_confidence_threshold=threshold_human, 
                                                                           head_iou_threshold=head_iou_threshold, 
                                                                           head_pred_confidence_threshold=threshold_head
                                                                        )
            
            ## add the new keys and update the results for an image
            image_result.update(result_dict1)
            if printFlag: print(f"\n\nresult_dict1 : {result_dict1}")            
            
            ## update the metrics for categories
            for label in required_classes:
                class_metrics[label]["precision"].append(result_dict2["category_performance"][label]["precision"])
                class_metrics[label]["recall"].append(result_dict2["category_performance"][label]["recall"])
                class_metrics[label]["average_iou"].append(result_dict2["category_performance"][label]["average_iou"])
                class_metrics[label]["tp"]+=(result_dict2["category_performance"][label]["tp"])
                class_metrics[label]["fp"]+=(result_dict2["category_performance"][label]["fp"])
                class_metrics[label]["fn"]+=(result_dict2["category_performance"][label]["fn"])

            if printFlag: print("\n\ncategory_performance : \n", class_metrics)

            # update results in the final dictionary for an image
            imagewise_results.append(image_result)
            if printFlag:
                click.secho("\n--------- RESULTS -------- : \n")
                for item in image_result.items(): print(item)
                # print("-"*150, "\n")

            ######## for loop ends here           

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            line_number = exc_tb.tb_lineno
            click.secho(f"Exception: {e} ----->> Error Type: {exc_type} ----->> File Name: {fname} ----->> Line Number: {line_number}", fg="red")

    try:
        print("\nsaveExcelFlag : ", saveExcelFlag)
        if saveExcelFlag:
            ######## Save image-wise performance excel report
            imagewise_df = pd.DataFrame(imagewise_results)
            click.secho(f"\n\nProcessed image files evaluation results : {imagewise_df.shape} ", fg="blue")
            save_csv_to_dbfs(imagewise_df, outputImgwiseExcel)

            ######## Calculate image-wise and overall model performance & save excel
            class_metrics_result, overall_metrics = calculate_model_performance(class_metrics)
            result_df = get_model_performance_dataframe(class_metrics_result, overall_metrics)
            save_csv_to_dbfs(result_df, outputOverallExcel)
            print("\n**************** Model Performance ****************\n", result_df.head())
            if printFlag : print("\nclass_metrics_result:", class_metrics_result, "\noverall_metrics : ",overall_metrics)
            
    except Exception as e:
        click.secho(f"", fg="red")
