import argparse
import os
import numpy as np
import SimpleITK as sitk
import json

### DO NOT add any additional imports or the autograder may fail ###


def get_num_cells(image_path, kernel_path):
    # TODO: Convert the blood image at the given "image_path" to a gray scale SITK image (use the green RGB channel)
    #       Convert the kernel image at the given "kernel_path" to an Otsu thresholded SITK Image
    #       Apply a convolution on the blood cell image using the thresholded kernel
    #       Apply Otsu's method on the result of the convolution
    #       Count the number of connected components in the the Otsu tresholded image
    #       Return the count
    image = sitk.ReadImage(image_path, sitk.sitkVectorUInt8)
    green_channel = sitk.VectorIndexSelectionCast(image, 1)

    kernel = sitk.ReadImage(kernel_path, sitk.sitkVectorUInt8)
    red_channel = sitk.VectorIndexSelectionCast(kernel, 0)

    kernel_otsu = sitk.OtsuThreshold(red_channel, 1, 0)

    green_float = sitk.Cast(green_channel, sitk.sitkFloat32)
    kernel_float = sitk.Cast(kernel_otsu, sitk.sitkFloat32)

    result = sitk.Convolution(green_float, kernel_float)
    result_otsu = sitk.OtsuThreshold(result, 1, 0)

    # get connected components and relabel to count
    filter = sitk.ConnectedComponentImageFilter()
    connected = filter.Execute(result_otsu)

    relabeled = sitk.RelabelComponentImageFilter()
    relabeled.Execute(connected)
    count = relabeled.GetNumberOfObjects()

    return count


def main(image_paths, kernel_paths):
    """
    DO NOT EDIT THIS CODE
    """
    ans_dict = {}

    #Iterate over all image files and all kernels
    for image_path in image_paths:
        for kernel_path in kernel_paths:
            count = get_num_cells(image_path, kernel_path)

            image = clean_file_name(image_path)
            kernel = clean_file_name(kernel_path)
            
            ans_dict[image] = {**ans_dict.get(image, {}), **{kernel: count}}

    print(ans_dict)

    # Write to the JSON file
    file_path = "blood_count.json"
    with open(file_path, 'w') as json_file:
        json.dump(ans_dict, json_file)
        

def get_files_from_folder(folder_path):
    """
    DO NOT EDIT THIS CODE
    """
    try:
        # Use os.listdir to get a list of filenames in the specified folder
        files = [f"{folder_path}/{f}" for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        return files
    except FileNotFoundError:
        print(f"Folder not found: {folder_path}")
        return []
    

def clean_file_name(path):
    """
    DO NOT EDIT THIS CODE
    """
    last_slash_index = path.rfind('/')
    filename = path[last_slash_index+1:]
    return filename
    
    
if __name__ == "__main__":
    """
    DO NOT EDIT THIS CODE
    """

    #Specify name of folder containing image files
    parser = argparse.ArgumentParser(description="Embedding Arguments")
    parser.add_argument(
        "-i", "--images", help="Path to image folder", required=True
    )

    parser.add_argument(
        "-k", "--kernels", help="Path to kernel folder", required=True
    )

    args = parser.parse_args()
    image_path = args.images

    kernel_path = args.kernels

    #Aquire names of image files
    image_paths = get_files_from_folder(image_path)
    kernel_paths = get_files_from_folder(kernel_path)

    #Pass image files to main
    main(image_paths, kernel_paths)
