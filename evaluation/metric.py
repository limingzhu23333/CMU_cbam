import cv2
import numpy as np
from scipy.spatial.distance import directed_hausdorff
import os

def dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred))

def hausdorff_distance(y_true, y_pred):
    distance_true = directed_hausdorff(y_true, y_pred)[0]
    distance_pred = directed_hausdorff(y_pred, y_true)[0]
    return max(distance_true, distance_pred)

def calculate_metrics(true_folder, pred_folder):
    results = []

    for file_name in os.listdir(true_folder):
        true_path = os.path.join(true_folder, file_name)
        pred_path = os.path.join(pred_folder, file_name)

        true_image = cv2.imread(true_path, cv2.IMREAD_GRAYSCALE)
        pred_image = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        
        if pred_image is None:
            # print(f"Warning: Prediction image not found for {file_name}")
            continue
        
        # Resize
        pred_image = cv2.resize(pred_image, (true_image.shape[1], true_image.shape[0]))

        # 
        true_image = np.where(true_image > 127, 1, 0).astype(np.uint8)
        pred_image = np.where(pred_image > 127, 1, 0).astype(np.uint8)

        # 
        dice = dice_coefficient(true_image, pred_image)

        # 
        hd = hausdorff_distance(true_image, pred_image)

        results.append((file_name, dice, hd))

    return results

# 
true_folder = 'D:/dose/result/label_segmentation_images/'  # 
pred_folder = 'D:/dose/result/predicted_segmentation_images/'    # 

# 
results = calculate_metrics(true_folder, pred_folder)

# 
with open('D:/dose/result/results.txt', 'w') as file:
    for file_name, dice, hd in results:
        
        file.write(f"{file_name}\t{dice}\t{hd}\n")

# 
with open('D:/dose/result/results.txt', 'r') as file:
    lines = file.readlines()

# 
with open('D:/dose/result/selected_results.txt', 'w') as file:
    file.writelines(selected_lines)

# 
with open('D:/dose/result/selected_results.txt', 'r') as file:
    lines = file.readlines()

# 
selected_lines = []
for line in lines:
    file_name = line.split('\t')[0]  # 
    # 
    first_three_digits = int(file_name[:3])
    # 
    if first_three_digits <= 100:
        selected_lines.append(line)

# 
with open('D:/dose/result/selected_results_loss.txt', 'w') as file:
    file.writelines(selected_lines)

import numpy as np

#
with open('D:/dose/result/selected_results_loss.txt', 'r') as file:
    lines = file.readlines()

# 
grouped_results = {}

# 
for line in lines:
    file_name, dice, hd = line.strip().split('\t')
    first_three_digits = file_name[:3]
    if first_three_digits not in grouped_results:
        grouped_results[first_three_digits] = []
    grouped_results[first_three_digits].append((float(dice), float(hd)))

# 
with open('D:/dose/result/selected_results_loss_mean.txt', 'w') as file:
    for group, results in grouped_results.items():
        dice_scores, hd_distances = zip(*results)
        avg_dice = np.mean(dice_scores)
        std_dice = np.std(dice_scores)
        avg_hd = np.mean(hd_distances)
        std_hd = np.std(hd_distances)
        file.write(f"Group: {group}, Average Dice: {avg_dice}, Standard Deviation Dice: {std_dice}, Average HD: {avg_hd}, Standard Deviation HD: {std_hd}\n")



import cv2
import numpy as np
import os

def calculate_sensitivity_specificity(true_folder, pred_folder, output_file):
    with open(output_file, 'w') as file:
        file.write("Image Name\tSensitivity\tSpecificity\n")
        
        true_files = os.listdir(true_folder)
        pred_files = os.listdir(pred_folder)

        for file_name in true_files:
            if file_name in pred_files:
                true_path = os.path.join(true_folder, file_name)
                pred_path = os.path.join(pred_folder, file_name)

                true_image = cv2.imread(true_path, cv2.IMREAD_GRAYSCALE)
                pred_image = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

                # 
                true_binary = np.where(true_image > 127, 1, 0).astype(np.uint8)
                pred_binary = np.where(pred_image > 127, 1, 0).astype(np.uint8)

                # 
                TP = np.sum(np.logical_and(true_binary == 1, pred_binary == 1))
                FN = np.sum(np.logical_and(true_binary == 1, pred_binary == 0))
                TN = np.sum(np.logical_and(true_binary == 0, pred_binary == 0))
                FP = np.sum(np.logical_and(true_binary == 0, pred_binary == 1))

                # 
                sensitivity = TP / (TP + FN)
                specificity = TN / (TN + FP)

                file.write(f"{file_name}\t{sensitivity:.4f}\t{specificity:.4f}\n")
            else:
                print(f"Warning: Prediction image not found for {file_name}")

# 
true_folder = 'E:/CT_Seg/label_segmentation_images/'  # 
pred_folder = 'E:/CT_Seg/predicted_segmentation_images/'    # 
output_file = 'E:/CT_Seg/sensitivity_specificity_results.txt'

# 
calculate_sensitivity_specificity(true_folder, pred_folder, output_file)

import numpy as np

# 
with open('E:/CT_Seg/sensitivity_specificity_results.txt', 'r') as file:
    lines = file.readlines()[1:]  # 

# 
grouped_results = {}

# 
for line in lines:
    file_name, sensitivity, specificity = line.strip().split('\t')
    first_three_digits = file_name[:3]
    if first_three_digits not in grouped_results:
        grouped_results[first_three_digits] = []
    grouped_results[first_three_digits].append((float(sensitivity), float(specificity)))

# 
with open('E:/CT_Seg/sensitivity_specificity_results_mean.txt', 'w') as file:
    for group, results in grouped_results.items():
        sensitivity, specificity = zip(*results)
        avg_sensitivity = np.mean(sensitivity)
        std_sensitivity = np.std(sensitivity)
        avg_specificity = np.mean(specificity)
        std_specificity = np.std(specificity)
        file.write(f"Group: {group}, Average sensitivity: {avg_sensitivity}, Standard Deviation sensitivity: {std_sensitivity}, Average specificity: {avg_specificity}, Standard Deviation specificity: {std_specificity}\n")
