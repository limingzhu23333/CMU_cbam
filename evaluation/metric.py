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
        
        # Resize预测图像以匹配真实图像的大小
        pred_image = cv2.resize(pred_image, (true_image.shape[1], true_image.shape[0]))

        # 将图像转换为二值图像（0或1）
        true_image = np.where(true_image > 127, 1, 0).astype(np.uint8)
        pred_image = np.where(pred_image > 127, 1, 0).astype(np.uint8)

        # 计算 Dice 系数
        dice = dice_coefficient(true_image, pred_image)

        # 计算 Hausdorff 距离
        hd = hausdorff_distance(true_image, pred_image)

        results.append((file_name, dice, hd))

    return results

# 文件夹路径
true_folder = 'D:/dose/result/label_segmentation_images/'  # 真实分割图像文件夹路径
pred_folder = 'D:/dose/result/predicted_segmentation_images/'    # 预测分割图像文件夹路径

# 计算指标
results = calculate_metrics(true_folder, pred_folder)

# 将结果写入txt文件
with open('D:/dose/result/results.txt', 'w') as file:
    for file_name, dice, hd in results:
        
        file.write(f"{file_name}\t{dice}\t{hd}\n")

# 读取结果文件
with open('D:/dose/result/results.txt', 'r') as file:
    lines = file.readlines()

# 将选取的结果写入新的 txt 文件
with open('D:/dose/result/selected_results.txt', 'w') as file:
    file.writelines(selected_lines)

# 读取结果文件
with open('D:/dose/result/selected_results.txt', 'r') as file:
    lines = file.readlines()

# 筛选文件名前三位数字大于100的行
selected_lines = []
for line in lines:
    file_name = line.split('\t')[0]  # 获取文件名
    # 提取文件名前三位数字并转换为整数
    first_three_digits = int(file_name[:3])
    # 如果前三位数字小于等于100，则将该行添加到选取的结果中
    if first_three_digits <= 100:
        selected_lines.append(line)

# 将选取的结果写入新的 txt 文件
with open('D:/dose/result/selected_results_loss.txt', 'w') as file:
    file.writelines(selected_lines)

import numpy as np

# 读取结果文件
with open('D:/dose/result/selected_results_loss.txt', 'r') as file:
    lines = file.readlines()

# 创建字典，用于存储文件名前三个数字相同的行
grouped_results = {}

# 将结果按照文件名前三个数字进行分组
for line in lines:
    file_name, dice, hd = line.strip().split('\t')
    first_three_digits = file_name[:3]
    if first_three_digits not in grouped_results:
        grouped_results[first_three_digits] = []
    grouped_results[first_three_digits].append((float(dice), float(hd)))

# 计算每组的平均值和标准差，并将结果写入新的 txt 文件
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

                # 将图像转换为二值图像（0或1）
                true_binary = np.where(true_image > 127, 1, 0).astype(np.uint8)
                pred_binary = np.where(pred_image > 127, 1, 0).astype(np.uint8)

                # 计算真阳性（TP）、假阴性（FN）、真阴性（TN）、假阳性（FP）
                TP = np.sum(np.logical_and(true_binary == 1, pred_binary == 1))
                FN = np.sum(np.logical_and(true_binary == 1, pred_binary == 0))
                TN = np.sum(np.logical_and(true_binary == 0, pred_binary == 0))
                FP = np.sum(np.logical_and(true_binary == 0, pred_binary == 1))

                # 计算敏感性（Sensitivity）和特异性（Specificity）
                sensitivity = TP / (TP + FN)
                specificity = TN / (TN + FP)

                file.write(f"{file_name}\t{sensitivity:.4f}\t{specificity:.4f}\n")
            else:
                print(f"Warning: Prediction image not found for {file_name}")

# 文件夹路径
true_folder = 'E:/CT_Seg/label_segmentation_images/'  # 真实分割图像文件夹路径
pred_folder = 'E:/CT_Seg/predicted_segmentation_images/'    # 预测分割图像文件夹路径
output_file = 'E:/CT_Seg/sensitivity_specificity_results.txt'

# 计算敏感性和特异性
calculate_sensitivity_specificity(true_folder, pred_folder, output_file)

import numpy as np

# 读取结果文件
with open('E:/CT_Seg/sensitivity_specificity_results.txt', 'r') as file:
    lines = file.readlines()[1:]  # 跳过标题行

# 创建字典，用于存储文件名前三个数字相同的行
grouped_results = {}

# 将结果按照文件名前三个数字进行分组
for line in lines:
    file_name, sensitivity, specificity = line.strip().split('\t')
    first_three_digits = file_name[:3]
    if first_three_digits not in grouped_results:
        grouped_results[first_three_digits] = []
    grouped_results[first_three_digits].append((float(sensitivity), float(specificity)))

# 计算每组的平均值和标准差，并将结果写入新的 txt 文件
with open('E:/CT_Seg/sensitivity_specificity_results_mean.txt', 'w') as file:
    for group, results in grouped_results.items():
        sensitivity, specificity = zip(*results)
        avg_sensitivity = np.mean(sensitivity)
        std_sensitivity = np.std(sensitivity)
        avg_specificity = np.mean(specificity)
        std_specificity = np.std(specificity)
        file.write(f"Group: {group}, Average sensitivity: {avg_sensitivity}, Standard Deviation sensitivity: {std_sensitivity}, Average specificity: {avg_specificity}, Standard Deviation specificity: {std_specificity}\n")
