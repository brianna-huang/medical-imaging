import SimpleITK as sitk
import numpy as np
import json

### DO NOT add any additional imports or the autograder may fail ###
### DO NOT change the method signatures ###

# DO NOT EDIT THIS FUCTION
def calculate_confusion_matrix(thresholded_image_mask, ground_truth_annotation_mask):
    """
    Function to calculate the number of true positives, false positives, true negatives, and false negatives
    for a given thresholded image mask and ground truth annotation mask.

    Args:
        thresholded_image_mask: a numpy array of 0s and 1s representing the predicted threshold mask 
        ground_truth_annotation_mask: a numpy array of 0s and 1s representing the ground truth annotation mask
    
    Returns: a tuple containing the number of true positives, false positives, true negatives, and false negatives
    """
    #True Positives
    TP = np.sum((thresholded_image_mask == 1) & (ground_truth_annotation_mask == 1))

    #False Positives
    FP = np.sum((thresholded_image_mask == 1) & (ground_truth_annotation_mask == 0))

    #True Negatives
    TN = np.sum((thresholded_image_mask == 0) & (ground_truth_annotation_mask == 0))

    #False Negatives
    FN = np.sum((thresholded_image_mask == 0) & (ground_truth_annotation_mask == 1))
    
    return TP, FP, TN, FN


def evaluateImageThresholdSegmentation(image, threshold, imageAnnotation):
    """
    Threshold the image and evaluate the quality of the segmentation
    by comparing the segmented image against a ground truth annotation.
    Compute the metrics to assess the quality of the segmentation.
    
    Args:
        image: a Simple ITK image with a single gray value for each pixel
        threshold: a given threshold value to segment the image
        imageAnnotation: a Simple ITK image representing the ground truth
            segmentation mask where each pixel is an unsigned 8-bit integer

    Returns: DICE, Jaccard Index, Sensitivity, Positive Predictive Value
    """ 

    # TODO: Derive DICE, Jaccard Index, Sensitivity, and Positive Predictive Value from 
    #       True Positives, False Positives, True Negatives, and False Negatives
    #       Ex: Sensitivity = TP/ (TP + FN)
    # NOTE: Pay attention to possible divide-by-zero errors. 
    #		For these metrics, define the metric as 0 when the denominator is zero.
    
    threshold_filter = sitk.BinaryThresholdImageFilter()
    threshold_filter.SetLowerThreshold(threshold)
    threshold_filter.SetUpperThreshold(255)
    threshold_filter.SetInsideValue(1)
    threshold_filter.SetOutsideValue(0)
    image_threshold = threshold_filter.Execute(image)

    thresholded_image = sitk.GetArrayFromImage(image_threshold).astype(np.uint8)
    true_image = sitk.GetArrayFromImage(imageAnnotation)

    TP, FP, TN, FN = calculate_confusion_matrix(thresholded_image, true_image)

    dice_div = 2 * TP + FP + FN
    dice = 2 * TP / dice_div if dice_div != 0 else 0
    jaccard_div = TP + FP + FN
    jaccard = TP / jaccard_div if jaccard_div != 0 else 0
    sens_div = TP + FN
    sensitivity = TP / sens_div if sens_div != 0 else 0
    ppv_div = TP + FP
    ppv = TP / ppv_div if ppv_div != 0 else 0

    return dice, jaccard, sensitivity, ppv


def bestThresholdBasedSegmentation(image, imageAnnotation, imageName, outFile):
    """
    Evaluate image results for all possible image threshold values.
    Identify the optimal threshold for each metric.
    Store the best evaluation metrics and their respective thresholds in resultDictionary.

    Args:
        image: a Simple ITK image with a single gray value for each pixel
        imageAnnotation: a Simple ITK image representing the ground truth
            segmentation mask where each pixel is an unsigned 8-bit integer
        imageName: a string - either "image1" or "image2" for the given image
        outFile: a json file where the results are written; 
    """   

    # Write the results to a json file
    # NOTE: This is the file the autograder checks after running your code on an unseen image
    # TODO: Replace each None with the appropriate value

    # dictionary to keep best (metric value, threshold value)
    metrics = {
        'dice': (0, 0),
        'jaccard': (0, 0),
        'sensitivity': (0, 0),
        'ppv': (0, 0)
    }

    for threshold in range(256):
        dice, jaccard, sensitivity, ppv = evaluateImageThresholdSegmentation(image, threshold, imageAnnotation)

        if dice > metrics['dice'][0]:
            metrics['dice'] = (dice, threshold)
        if jaccard > metrics['jaccard'][0]:
            metrics['jaccard'] = (jaccard, threshold)
        if sensitivity > metrics['sensitivity'][0]:
            metrics['sensitivity'] = (sensitivity, threshold)
        if ppv > metrics['ppv'][0]:
            metrics['ppv'] = (ppv, threshold)
    
    resultDictionary = {'image' : imageName, 
                        'best DICE' : metrics['dice'][0], 
                        'best DICE threshold' : metrics['dice'][1],
                        'best Jaccard Index' : metrics['jaccard'][0], 
                        'best Jaccard Index threshold': metrics['jaccard'][1],
                        'best sensitivity' : metrics['sensitivity'][0], 
                        'best sensitivity threshold' : metrics['sensitivity'][1],
                        'best positive predictive value' : metrics['ppv'][0], 
                        'best positive predictive value threshold' : metrics['ppv'][1]}
    json.dump(resultDictionary, outFile)

def get_seg(image, threshold, filename):
    """
    get the segmentation of an image as an mha file for viewing in Fiji!
    """
    thresh_filter = sitk.BinaryThresholdImageFilter()
    thresh_filter.SetLowerThreshold(threshold)
    thresh_filter.SetUpperThreshold(255)
    thresh_filter.SetInsideValue(255)
    thresh_filter.SetOutsideValue(0)
    
    segmentation = thresh_filter.Execute(image)
    sitk.WriteImage(segmentation, filename)


def main():
    """
    NOTE: There is flexibility in how you implement the main function. This is just a suggestion based on the \
        Detailed Steps section of the homework write-up.

    TODO: 1. Read the first data image and format it according to the instructions

    TODO: 2. Read the corresponding annotation mask and format it according to the instructions

    TODO: 3. Calculate the metrics for image1

    TODO: 4. Repeat steps 1-3 for the second image

    NOTE: To use evaluateImageThresholdSegmentation() on image2, set the threshold argument to the best DICE threshold \
        value calculated from running bestThresholdBasedSegmentation() on image1.
    """

    # read in images and annotations
    image1 = sitk.ReadImage("XRay-Knee1.jpg", sitk.sitkUInt8)
    image1 = sitk.Extract(image1, (image1.GetSize()[0], image1.GetSize()[1], 0))

    annotation1 = sitk.ReadImage("XRay-Knee1-seg.mha", sitk.sitkUInt8)
    annotation1 = sitk.Extract(annotation1, (annotation1.GetSize()[0], annotation1.GetSize()[1], 0))

    image2 = sitk.ReadImage("XRay-Knee2.png", sitk.sitkUInt8)
    image2 = sitk.Extract(image2, (image2.GetSize()[0], image2.GetSize()[1], 0))

    annotation2 = sitk.ReadImage("XRay-Knee2-seg.mha", sitk.sitkUInt8)
    annotation2 = sitk.Extract(annotation2, (annotation2.GetSize()[0], annotation2.GetSize()[1], 0))

    # question 1
    q1_results = evaluateImageThresholdSegmentation(image1, 126, annotation1)
    print("q1 results:")
    print(q1_results)
    print()

    # question 2
    with open("q2.json", "w") as output:
        bestThresholdBasedSegmentation(image1, annotation1, "image1", output)
    print("q2 results:")
    with open('q2.json', 'r') as f:
        data = json.load(f)
        print(json.dumps(data, indent=4))
    print()

    # question 3
    with open("q2.json", "r") as f:
        image1_results = json.load(f)
    best_dice_threshold = image1_results['best DICE threshold']
    q3_results = [best_dice_threshold]
    q3_results.append(evaluateImageThresholdSegmentation(image2, best_dice_threshold, annotation2))
    print("q3 results:")
    print(q3_results)
    print()

    # question 4
    with open("q4.json", "w") as output:
        bestThresholdBasedSegmentation(image2, annotation2, "image2", output)
    print("q4 results:")
    with open('q4.json', 'r') as f:
        data = json.load(f)
        print(json.dumps(data, indent=4))

    # question 5
    get_seg(image1, 116, "dice_seg.mha")
    get_seg(image1, 162, "ppv_seg.mha")
    get_seg(image1, 0, "sensitivity_seg.mha")


if __name__ == "__main__":
    main()