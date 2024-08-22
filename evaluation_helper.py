
import numpy as np, os, sys, click, cv2, pandas as pd
import xml.etree.ElementTree as ET
# from visualize_predicted_bounding_boxes import (draw_bounding_boxes_with_legend)
from pprint import pprint

def parse_annotation(xml_tree, required_classes):
    try:
        annotations = []
        if xml_tree is not None:
            root = xml_tree.getroot()
            # tree = ET.parse(xml_file)
            # root = tree.getroot()
            for obj in root.findall('object'):
                category = obj.find('name').text
                if category in required_classes:
                    bbox = obj.find('bndbox')
                    xmin = int(bbox.find('xmin').text)
                    ymin = int(bbox.find('ymin').text)
                    xmax = int(bbox.find('xmax').text)
                    ymax = int(bbox.find('ymax').text)
                    annotations.append({"bbox": [xmin, ymin, xmax, ymax],
                                        "category": category,
                                        "score": 1.0})
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        click.secho(f"Exception: {e}\nError Type: {exc_type}\nFile Name: {fname}\nLine Number: {exc_tb.tb_lineno}", fg="red")
    return annotations
    
def calculate_iou(bbox1, bbox2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    Parameters:
    bbox1 (list or tuple): The coordinates of the first bounding box in the format [xmin, ymin, xmax, ymax].
    bbox2 (list or tuple): The coordinates of the second bounding box in the format [xmin, ymin, xmax, ymax].
    Returns:
    float: The IoU value.
    """
    try:
        iou=0
        # Determine the (x, y)-coordinates of the intersection rectangle
        x_left = max(bbox1[0], bbox2[0])
        y_top = max(bbox1[1], bbox2[1])
        x_right = min(bbox1[2], bbox2[2])
        y_bottom = min(bbox1[3], bbox2[3])
        # Calculate the area of intersection rectangle
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        # Calculate the area of both bounding boxes
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        # Calculate the union area
        union_area = bbox1_area + bbox2_area - intersection_area
        # Compute the IoU
        iou = intersection_area / union_area
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        click.secho(f"Exception: {e}\nError Type: {exc_type}\nFile Name: {fname}\nLine Number: {exc_tb.tb_lineno}", fg="red")
    return iou


def calculate_class_wise_metric(required_classes, matches, pred_class_counts, gt_class_counts):
    try:
        # store class wise results
        class_metrics = {}
        # overall metrics results
        overall_tp = 0
        overall_fp = 0
        overall_fn = 0
        # average precision
        aps = []
        # iterate class wise
        for category in required_classes:
            tp = len(matches[category])
            fp = pred_class_counts[category] - tp
            fn = gt_class_counts[category] - tp
            overall_tp += tp
            overall_fp += fp
            overall_fn += fn
            # precision recall for a class
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            average_iou = sum(calculate_iou(gt["bbox"], pred["bbox"]) for gt, pred in matches[category]) / tp if tp > 0 else 0
            # define results for a class
            class_metrics[category] = {
                "precision": round(precision, 2),
                "recall": round(recall, 2),
                "average_iou": round(average_iou, 2),
                "tp": tp,
                "fp": fp,
                "fn": fn
            }

        # Calculate overall precision and recall for an image
        overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0
        overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
        # Calculate overall mAP
        mAP = sum(aps) / len(aps) if aps else 0.0
        result = {"mAP": round(mAP, 2),
                  "overall_precision": round(overall_precision, 2),
                  "overall_recall": round(overall_recall, 2),
                  "overall_tp":overall_tp,
                  "overall_fp":overall_fp,
                  "overall_fn":overall_fn,
                  "category": class_metrics}

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        click.secho(f"Exception: {e}\nError Type: {exc_type}\nFile Name: {fname}\nLine Number: {exc_tb.tb_lineno}", fg="red")
    return result

def calculate_ap(precision, recall):
    """
    Calculate Average Precision (AP) given lists of precision and recall values.
    """
    if len(precision) == 1 and len(recall) == 1:
        return precision[0] * recall[0]
    # Ensure precision and recall are sorted in ascending order of recall
    recall, precision = zip(*sorted(zip(recall, precision)))
    # Append sentinel values at the end
    recall = [0.0] + list(recall) + [1.0]
    precision = [0.0] + list(precision) + [0.0]
    # Compute the precision envelope
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    # Integrate area under precision-recall curve
    ap = 0.0
    for i in range(1, len(recall)):
        ap += (recall[i] - recall[i - 1]) * precision[i]
    return ap


def match_predictions_to_ground_truth(
    ground_truth, 
    predictions, 
    required_classes, 
    human_iou_threshold=0.5, 
    human_pred_confidence_threshold=0.5, 
    head_iou_threshold=0.01, 
    head_pred_confidence_threshold=0.3
):
    ## metrics 
    printTrue = False
    image_metrics = {} # represents overall precision, recall for an image
    class_wise_metric = {} # represents category wise precision, recall for an image
    unmatched_predictions={}
    gt_class_counts = {}
    pred_class_counts = {}
    matches = {}
    used_gt_indices = {}
    
    # define

    for label in required_classes:
        # GT category count
        gt_class_counts[label]=0
        # Prediction category count
        pred_class_counts[label]=0
        # store the matching pairs of bbox GT & Pred pairs
        ## define params for best pair extraction
        # store the matching bbox GT & Pred pairs
        matches[label] = []
        # matches = {"standing": [], "head": [], "crouching": []}
        # store the unmatched bbox GT & Pred pairs
        unmatched_predictions[label] = []
        # unmatched_predictions = {"standing": [], "head": [], "crouching": []}
        # store index of the used best matching bbox 
        used_gt_indices[label] = set()
        # used_gt_indices = {"standing": set(), "head": set(), "crouching": set()}

    # print(f"\nused_gt_indices : {used_gt_indices}\nunmatched_predictions: {unmatched_predictions}\nmatches : {matches}")
    # print(f"gt_class_counts : {gt_class_counts}\npred_class_counts : {pred_class_counts}\n")

    ## iterate over ground truths
    for gt in ground_truth:
        if printTrue:
            print("-"*100)
            click.secho(f"GT picked : {gt} --> {gt['category']}", fg="green")

        # update GT category count
        if gt["category"] in gt_class_counts:
            gt_class_counts[gt["category"]] += 1
        if gt["category"] == "head":
            iou_threshold = head_iou_threshold
            pred_confidence_threshold = head_pred_confidence_threshold
        else:
            iou_threshold = human_iou_threshold
            pred_confidence_threshold = human_pred_confidence_threshold

        # define params for best match of GT & Pred pairing
        best_iou = 0
        best_pred = None
        best_pred_index = -1

        # iterate over predictions
        for pred_index, pred in enumerate(predictions):    
            if printTrue : click.secho(f"Pred bbox picked : {pred} --> {pred['category']}, {pred['score']}", fg="blue")
            
            # if prediction score is good & GT-Pred category matches
        

            if pred["score"] >= pred_confidence_threshold and pred["category"] == gt["category"]:
                # calculate iou
                iou = calculate_iou(gt["bbox"], pred["bbox"])
                if printTrue : print(f"iou = {iou}")
                # update best iou
                if iou > best_iou:
                    best_iou = iou
                    best_pred = pred
                    best_pred_index = pred_index
                    if printTrue : print(f"iou > best_iou --> iou : {iou}, best_iou : {best_iou}, best_pred_index: {best_pred_index}")
            else:
                if printTrue: print('pred["category"] != gt["category"]')
        
        # if the category is head then set different threshold
    

        # consider match/pair of GT & pred if best_iou > iou_threshold
        # also, the best matching bbox is unused
        if best_iou >= iou_threshold and best_pred_index not in used_gt_indices[gt["category"]]:
            matches[gt["category"]].append((gt, best_pred))
            used_gt_indices[gt["category"]].add(best_pred_index)
            if printTrue:
                print(f"\nbest_iou >= iou_threshold and best_pred_index not in used_gt_indices ---->")
                print(f"matches.append : {(gt, best_pred)}")
                print(f"used_gt_indices.append : {best_pred_index}")
        else:
            unmatched_predictions[gt["category"]].append(best_pred)
            if printTrue : print(f"\nunmatched_predictions.append : {best_pred}")
    
    # loop ends here

    # count the categories in prediction
    for pred in predictions:
        if pred["category"] in pred_class_counts:
            pred_class_counts[pred["category"]] += 1

    # get evaluation metrics category wise
    # calculate overall metric for an image & all the individual classes
    image_performance = calculate_class_wise_metric(required_classes,
                                                    matches,
                                                    pred_class_counts,
                                                    gt_class_counts)

    # print(f"\nused_gt_indices : {used_gt_indices}\nunmatched_predictions: {unmatched_predictions}\nmatches : {matches}")
    # print(f"gt_class_counts : {gt_class_counts}\npred_class_counts : {pred_class_counts}\n")
    
    if printTrue : 
        click.secho(f"Individual performance (calculate_class_wise_metric) : ", fg="yellow")
        pprint(image_performance)

    ##### set results
    result = {"Precision CORRECT_DETECTIONS": image_performance["overall_precision"],
             "Recall DETECTIONS_NOT_MISSED": image_performance["overall_recall"]}

    for label in required_classes:
        result[f"GT {label}"] = gt_class_counts[label]
        result[f"Pred {label}"] = pred_class_counts[label]
    
    for label in required_classes:
        result[f"Precision {label}"] = image_performance["category"][label]["precision"]
        result[f"Recall {label}"] = image_performance["category"][label]["recall"]
        result[f"Avg.IoU {label}"] = image_performance["category"][label]["average_iou"]

    result.update({"True Positive Correctly_Pred": image_performance["overall_tp"],
             "False Positive Incorrectly_Pred": image_performance["overall_fp"],
             "False Negative Failed_to_Detect": image_performance["overall_fn"],
            })

    outputs = {"GT_Pred_Pairing": matches, # pairs of GT & Pred bounding boxes
               "category_performance": image_performance["category"] # class wise performance metrics
            }
    return result, outputs


def compute_overall_model_performance(class_wise_metric):
    """
    Compute AP, Precision, and Recall for each class and mAP overall.
    Args:
    class_wise_metric: A dictionary with class names as keys and a dictionary of "precision", "recall", and "average_iou" lists as values.
    Returns:
    A dictionary with overall mAP, and metrics for each class.
    """
    class_results = {}
    overall_tp = 0
    overall_fp = 0
    overall_fn = 0
    overall_average_iou = 0.0
    # average precision
    aps = []
    # iterate
    for category, metrics in class_wise_metric.items():
        tp = metrics["tp"]
        fp = metrics["fp"]
        fn = metrics["fn"]
        overall_tp += tp
        overall_fp += fp
        overall_fn += fn
        precision = metrics["precision"]
        recall = metrics["recall"]
        # average precision
        ap = calculate_ap(precision, recall)
        # category wise metrics
        category_precision = sum(precision) / len(precision) if precision else 0 
        category_recall = sum(recall) / len(recall) if recall else 0
        category_average_iou = sum(metrics["average_iou"]) / len(metrics["average_iou"]) if metrics["average_iou"] else 0
        overall_average_iou+=category_average_iou

        class_results[category] = {
            "precision": round(category_precision, 2),
            "recall": round(category_recall, 2),
            "average_iou": round(category_average_iou, 2),
            "ap": round(ap, 2)
        }
        aps.append(ap)

    # Calculate overall precision and recall
    overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0
    overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
    overall_average_iou = round((overall_average_iou / len(class_wise_metric.items())) * 100, 2)

    # Calculate overall mAP
    mAP = sum(aps) / len(aps) if aps else 0.0
    return {
        "overall_precision": round(overall_precision, 2),
        "overall_recall": round(overall_recall, 2),
        "mAP": round(mAP, 2),
        "overall_tp": overall_tp,
        "overall_fp": overall_fp,
        "overall_fn": overall_fn,
        "overall_average_iou": overall_average_iou,
        "class_results": class_results
    }

def calculate_model_performance(category_performance):
    total_tp = total_fp = total_fn = 0
    class_wise_metrics = {}
    ap_list = []
    for category, metrics in category_performance.items():
        tp = metrics['tp']
        fp = metrics['fp']
        fn = metrics['fn']
        precision = sum(metrics['precision']) / len(metrics['precision']) if metrics['precision'] else 0
        recall = sum(metrics['recall']) / len(metrics['recall']) if metrics['recall'] else 0
        average_iou = sum(metrics['average_iou']) / len(metrics['average_iou']) if metrics['average_iou'] else 0
        class_wise_metrics[category] = {
            'precision': round(precision,2),
            'recall': round(recall,2),
            'average_iou': round(average_iou,2),
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
        ap_list.append(precision)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    accuracy = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0
    mAP = sum(ap_list) / len(ap_list) if ap_list else 0
    overall_metrics = {
        'overall_precision': round(overall_precision, 2),
        'overall_recall': round(overall_recall, 2),
        'accuracy': round(accuracy, 2),
        'mAP': round(mAP, 2)
    }
    return class_wise_metrics, overall_metrics
    
def get_model_performance_dataframe(category_performance, overall_performance):
    # Combine the dictionaries
    combined_data = []
    # Add category performance
    for category, metrics in category_performance.items():
        metrics['category'] = category
        combined_data.append(metrics)

    # Add overall performance
    overall_metrics = {'category': 'overall'}
    overall_metrics.update(overall_performance)
    combined_data.append(overall_metrics)
    # Convert the combined data to a DataFrame
    df = pd.DataFrame(combined_data)
    # Fill NaN values with empty strings for better readability
    df = df.fillna('')
    return df

def save_overall_results_to_an_excel(performance_dict, output_path='performance_metrics.xlsx'):
    try:
       # Prepare class results data
        class_data = performance_dict['class_results']
        class_df = pd.DataFrame(class_data).T.reset_index().rename(columns={'index': 'class'})
        # Prepare overall metrics data
        overall_data = {
            'class': ['overall'],
            'ap': [performance_dict['mAP']],
            'average_iou': [performance_dict['overall_average_iou']],
            'precision': [performance_dict['overall_precision']],
            'recall': [performance_dict['overall_recall']],
            'tp': [performance_dict['overall_tp']],
            'fp': [performance_dict['overall_fp']],
            'fn': [performance_dict['overall_fn']]
        }
        overall_df = pd.DataFrame(overall_data)
        # Merge class results and overall metrics
        result_df = pd.concat([class_df, overall_df], ignore_index=True)
        # Save to Excel
        result_df.to_excel(output_path, index=False)
        return True
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        click.secho(f"Exception: {e}\nError Type: {exc_type}\nFile Name: {fname}\nLine Number: {exc_tb.tb_lineno}", fg="red")
        return False


if __name__ == "__main__":
    # The performance dictionary
    performance_dict = {
        'class_results': {
            'crouching': {'ap': 0.0, 'average_iou': 0.0, 'precision': 0.0, 'recall': 0.0},
            'head': {'ap': 0.74, 'average_iou': 0.42, 'precision': 0.83, 'recall': 0.77},
            'standing': {'ap': 1.0, 'average_iou': 0.82, 'precision': 1.0, 'recall': 0.75}
        },
        'mAP': 0.58,
        'overall_average_iou': 41.33,
        'overall_fn': 4,
        'overall_fp': 2,
        'overall_precision': 0.87,
        'overall_recall': 0.76,
        'overall_tp': 13
    }


def calculate_general_metric(matches, predictions, ground_truth):
    metrics = {"precision":0.0,
                "recall" : 0.0,
                "average_iou": 0.0,
                "tp":0,
                "fp":0,
                "fn":0
            }
    tp = len(matches)
    fp = len(predictions) - tp
    fn = len(ground_truth) - tp
    # Compute Precision and Recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    # Compute Average IoU
    average_iou = sum(calculate_iou(gt["bbox"], pred["bbox"]) for gt, pred in matches[category]) / tp if tp > 0 else 0
    metrics["average_iou"] = round(average_iou, 2)
    metrics["precision"] = round(precision, 2)
    metrics["recall"] = round(recall, 2)
    metrics["tp"] = tp
    metrics["fp"] = fp
    metrics["fn"] = fn
    return metrics

def compute_overall_metrics(class_wise_metric):
    """
    Compute overall precision, recall, and mAP from class-wise metrics.
    
    Args:
    class_wise_metric: A dictionary with class names as keys and a dictionary of "precision", "recall", and "average_iou" lists as values.
    
    Returns:
    A dictionary with overall precision, recall, and mAP.
    """
    overall_tp = 0
    overall_fp = 0
    overall_fn = 0
    aps = {}
    # iterate over the categories
    for category, metrics in class_wise_metric.items():
        # get tp, fp, fn
        tp = metrics["tp"]
        fp = metrics["fp"]
        fn = metrics["fn"]
        # sum up the tp, fp, fn
        overall_tp += tp
        overall_fp += fp
        overall_fn += fn
        # extract precision recall values
        precision = metrics["precision"]
        recall = metrics["recall"]
        # calculate average precision
        ap = calculate_ap(precision, recall)
        aps[category] = ap
    # Calculate overall precision and recall
    overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0
    overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
    # Calculate overall mAP
    mAP = sum(aps.values()) / len(aps) if aps else 0.0
    return {
        "mAP": round(mAP, 2),
        "overall_precision": round(overall_precision, 2),
        "overall_recall": round(overall_recall, 2),
        "overall_tp":overall_tp,
        "overall_fp":overall_fp,
        "overall_fn":overall_fn
    }
