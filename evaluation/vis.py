import cv2
import matplotlib.pyplot as plt

# 读取 CT 图像、真实标签图像和预测分割图像
# ct_image = cv2.imread('E:/avm61-100/nii/png/images2/077_142.png')
ct_image = cv2.imread('F:/picture/png/098_2/098_2_169.png')
true_label_image = cv2.imread('D:/jpg_seg/jpg_seg/src/label_segmentation_images/098_169.png', cv2.IMREAD_GRAYSCALE)
predicted_segmentation_image = cv2.imread('D:/jpg_seg/jpg_seg/src/predicted_segmentation_images/098_169.png', cv2.IMREAD_GRAYSCALE)

overlay_image = ct_image.copy()

overlay_image[true_label_image == 255] = [0, 0, 255]  # 将真实标签区域设置为红色
# overlay_image[predicted_segmentation_image == 255] = [250, 140, 50]  # 将预测分割结果区域设置为蓝色

# 设置透明度
alpha = 0.7 # 透明度为 0.5

# 叠加图像
overlay = cv2.addWeighted(ct_image, alpha, overlay_image, 1 - alpha, 0)

# 可视化叠加后的图像
plt.figure(figsize=(10, 10))
# plt.imshow(cv2.cvtColor(ct_image, cv2.COLOR_BGR2RGBA))
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGRA2RGBA))
plt.axis('off')
plt.show()


import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取 CT 图像、真实标签图像和预测分割图像
ct_image = cv2.imread('F:/picture/png/075_3/075_3_150.png')
true_label_image = cv2.imread('D:/jpg_seg/jpg_seg/src/label_segmentation_images/075_150.png', cv2.IMREAD_GRAYSCALE)
predicted_segmentation_image = cv2.imread('D:/jpg_seg/jpg_seg/src/predicted_segmentation_images/075_150.png', cv2.IMREAD_GRAYSCALE)

# 创建彩色叠加图像
overlay_image = ct_image.copy()

# 对背景图像应用直方图均衡化并略微降低亮度
ct_image_hsv = cv2.cvtColor(ct_image, cv2.COLOR_BGR2HSV)
ct_image_hsv[:, :, 2] = cv2.equalizeHist(ct_image_hsv[:, :, 2])
ct_image_hsv[:, :, 2] = ct_image_hsv[:, :, 2] * 0.55  # 略微降低亮度
overlay_background = cv2.cvtColor(ct_image_hsv, cv2.COLOR_HSV2BGR)

# 使用Canny边缘检测器检测CT图像的边缘
edges = cv2.Canny(ct_image, 100, 200)

# 将检测到的边缘作为掩码，从原始CT图像中提取感兴趣区域
ct_image_filtered = cv2.bitwise_and(ct_image, ct_image, mask=edges)

# 设置透明度
alpha = 0.55 # 透明度为 0.7

# 叠加图像
overlay = cv2.addWeighted(overlay_background, alpha, overlay_image, 1 - alpha, 0)

# 计算两个图像的差值，并将差值图像设为黄色
diff_image = cv2.absdiff(true_label_image, predicted_segmentation_image)
diff_image = cv2.cvtColor(diff_image, cv2.COLOR_GRAY2BGR)
diff_image[np.all(diff_image == [255, 255, 255], axis=-1)] = [0, 230, 230]  # 将差值图像设为黄色

# 将差值图像叠加到原始图像上
overlay = cv2.addWeighted(overlay, 1, diff_image, 0.3, 0)

# 可视化叠加后的图像
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()


#dose
import pydicom
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from pathlib import Path
from dicompylercore import dicomparser

def read_dose(dose_path):
    ds_dose = pydicom.dcmread(dose_path)
    dose_array = ds_dose.pixel_array * ds_dose.DoseGridScaling
    dose_spacing = [float(each) for each in ds_dose.PixelSpacing] + [float(ds_dose.SliceThickness)]
    dose_origin = [float(each) for each in ds_dose.ImagePositionPatient]
    return dose_array, dose_spacing, dose_origin

def read_ct(ct_folder):
    reader = sitk.ImageSeriesReader()
    ct_files = reader.GetGDCMSeriesFileNames(ct_folder)
    reader.SetFileNames(ct_files)
    ct_image = reader.Execute()
    ct_array = sitk.GetArrayFromImage(ct_image)
    ct_spacing = ct_image.GetSpacing()
    ct_origin = ct_image.GetOrigin()
    ct_direction = ct_image.GetDirection()
    return ct_array, ct_spacing, ct_origin, ct_direction

def read_structure(structure_path, roi_number):
    structure = dicomparser.DicomParser(structure_path)
    structures = structure.GetStructures()
    roi = structures[roi_number]
    roi_planes = structure.GetStructureCoordinates(roi_number)
    return roi, roi_planes

def resample_dose_to_ct(dose_array, dose_spacing, dose_origin, ct_array, ct_spacing, ct_origin, ct_direction):
    # Create SimpleITK image from dose array
    dose_image = sitk.GetImageFromArray(dose_array)
    dose_image.SetSpacing(dose_spacing)
    dose_image.SetOrigin(dose_origin)
    
    # Define the resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(ct_spacing)
    resampler.SetOutputOrigin(ct_origin)
    resampler.SetSize(ct_array.shape[::-1])
    resampler.SetOutputDirection(ct_direction)
    resampler.SetInterpolator(sitk.sitkLinear)
    
    # Resample the dose image to match the CT image
    resampled_dose_image = resampler.Execute(dose_image)
    resampled_dose_array = sitk.GetArrayFromImage(resampled_dose_image)
    
    return resampled_dose_array

def plot_dose_and_structure(ct_array, dose_array, structure_planes, ct_origin, ct_spacing, slice_index):
    plt.figure(figsize=(10, 10))
    
    # Plot the CT array
    plt.imshow(ct_array[slice_index], cmap='gray', alpha=0.5)
    
    # Plot the dose array
    plt.imshow(dose_array[slice_index], cmap='jet', alpha=0.5)
    
    # Plot the structure contours
    z_coord = ct_origin[2] + slice_index * ct_spacing[2]
    for plane in structure_planes.values():
        for contour in plane:
            if np.isclose(contour['data'][0][2], z_coord, atol=ct_spacing[2] / 2):
                coords = np.array(contour['data'])
                plt.plot(coords[:, 0], coords[:, 1], 'r-', linewidth=1)
    
    plt.colorbar()
    plt.show()

def main():
    ct_folder = 'D:/dose/huangaiping/huangaiping_before/'
    dose_path = 'D:/dose/huangaiping/RT/1.2.840.114358.0382.2.20240510211533.39125207442_rtdose.dcm'
    structure_path = 'D:/dose/huangaiping/RT/1.2.840.114358.0382.2.20240510211524.39102944367_rtss.dcm'
    roi_number = 1  # Replace with the appropriate ROI number

    ct_array, ct_spacing, ct_origin, ct_direction = read_ct(ct_folder)
    dose_array, dose_spacing, dose_origin = read_dose(dose_path)
    structure, structure_planes = read_structure(structure_path, roi_number)

    resampled_dose_array = resample_dose_to_ct(dose_array, dose_spacing, dose_origin, ct_array, ct_spacing, ct_origin, ct_direction)
    print(resampled_dose_array.shape)
    
    slice_index = len(resampled_dose_array) // 2  # Choose a slice to plot
    plot_dose_and_structure(ct_array, resampled_dose_array, structure_planes, ct_origin, ct_spacing, slice_index)

if __name__ == "__main__":
    main()
