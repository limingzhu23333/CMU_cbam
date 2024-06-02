import SimpleITK as sitk
import glob
import os

p = list(range(0, 200))
for name in p:
    # Dicom save to nii
    
    # 使用 glob 获取匹配通配符的文件夹列表
    file_path = os.path.join("F:/data", f"{name}*/")
    matching_folders = glob.glob(file_path)

    # 检查是否有匹配的文件夹
    if not matching_folders:
        print(f"No matching folders found for {file_path}")
        continue

    # 取第一个匹配到的文件夹
    folder_path = matching_folders[0]

    # 获取该文件夹下的所有序列ID
    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(folder_path)
    if not series_IDs:
        print(f"No DICOM series found in {folder_path}")
        continue

    # 查看该文件夹下的序列数量
    nb_series = len(series_IDs)
    print(nb_series)
    

    # 通过ID获取该ID对应的序列所有切片的完整路径
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(folder_path, series_IDs[0])

    # 新建一个ImageSeriesReader对象
    series_reader = sitk.ImageSeriesReader()

    # 通过之前获取到的序列的切片路径来读取该序列
    series_reader.SetFileNames(series_file_names)

    # 获取该序列对应的3D图像
    image3D = series_reader.Execute()

    # 查看该3D图像的尺寸
    print(image3D.GetSize())

    # 将序列保存为单个的NRRD文件
    sitk.WriteImage(image3D, f'F:/data/nii/{name}_1.nii')


#读取靶区
import os
from dcmrtstruct2nii import dcmrtstruct2nii

# 定义输入路径和输出文件夹路径
path = 'D:/dose/99CHENGUANGMING/1.2.840.114358.0382.2.20240106183549.86651182322_rtss.dcm'
input_folder = 'D:/dose/99_1/'
output_folder = 'D:/dose/99_mask'  # 这里修改为你想要的输出文件夹路径

# 检查输出文件夹是否存在，如果不存在则自动创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 调用 dcmrtstruct2nii 函数
dcmrtstruct2nii(path, input_folder, output_folder)

import gzip

def decompress_nii_gz(nii_gz_path, output_path):
    with gzip.open(nii_gz_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            f_out.write(f_in.read())

# 示例使用
nii_gz_path = 'D:/dose/99_mask/mask_Tumor.nii.gz'
output_path = 'D:/dose/99_mask/mask_Tumor.nii'
decompress_nii_gz(nii_gz_path, output_path)



