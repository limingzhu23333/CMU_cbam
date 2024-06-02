import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

def normalization(data):
    assert isinstance(data, np.ndarray), "Input data should be a NumPy array"
    data = data.astype(np.float64)
    
    _min = np.min(data)
    _max = np.max(data)
    scaled_data = 255 * (data - _min) / (_max - _min)
    
    return scaled_data.astype(np.uint8)  # Convert back to uint8


# def normalization(data):
#     _range = np.max(data) - np.min(data)
#     return (data - np.min(data)) / (_range)

root = r"E:/avm1-30/avm1-30/nii/"

input_files = sorted([_file for _file in sorted(os.listdir(root)) if "nii" in _file])
i = 0

for patient_id, file_name in enumerate(input_files):
    file_name = file_name.split(".")[-2]
    # file_name = '00' + file_name
    # file_name = os.path.join(00,file_name)
    png_path = f"E:/avm1-30/avm1-30/seg_new/{file_name}/"
    os.makedirs(png_path, exist_ok=True)
    print(file_name, patient_id)
    file_name = str(file_name).zfill(3)
    input_filepath = os.path.join(root, input_files[patient_id])
    input_data = np.asanyarray(nib.load(input_filepath).get_fdata())
    print(input_data.shape)
    _a, _b, slices_num = input_data.shape
    input_data = normalization(input_data)
    for slice in range(slices_num):
        input_numpy = np.array(input_data[:, :, slice])
        # if np.max(input_numpy) == np.min(input_numpy):
        #     continue

        # input_numpy_resized = np.array(Image.fromarray(input_numpy))
        patient_id_pre = '%04d' % (patient_id + 1)
        prefix = '%04d' % i
        pic_path = os.path.join(png_path, file_name) + f"_{slice+1:03d}.png"
        
        img = Image.fromarray(input_numpy)
        if img.mode != 'L':
            img = img.convert('L')
        img = img.resize((512, 512), Image.ANTIALIAS)
        img.save(pic_path)
        i = i+1

print(f"Done! Total:{i} pairs")

import cv2
import glob
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image
# p = list(range(1,21))
p = [str(i).zfill(3) for i in range(31, 61)]
# p = str(p).zfill(3)

for name in p:
    input_img_domainA = glob.glob(f"F:/avm31-60/nii/png/{name}_1/*.png")
    file_names = [os.path.basename(file) for file in input_img_domainA]
    save_path = "F:/avm31-60/nii/png/new/"
    def mkdir(path):
        folder = os.path.exists(path)
        
        if not folder:  
            os.makedirs(path)  
            print("--- create new folder...  ---")
        else:
            print("---  There is this folder!  ---")
    mkdir(save_path)  

    i = 0
    for file in file_names:
        # print(file)
        file_name = "_".join(file.split("_")[:1] + [file.split("_")[-1]])
        # print(file_name)
        img1 = Image.open(f"F:/avm31-60/nii/png/{name}_1/{file}").convert("RGB")
        img1 = np.array(img1)
        # print(img1.shape)
        file_name_parts = file.split("_")
        file_name_parts[1] = "2" 
        file_name_120 = "_".join(file_name_parts[:2] + [file_name_parts[-1]])
        img2 = Image.open(f"F:/avm31-60/nii/png/{name}_2/{file_name_120}").convert("RGB")
        img2 = np.array(img2)
        file_name_parts1 = file.split("_")
        file_name_parts1[1] = "3" 
        file_name_3 = "_".join(file_name_parts1[:2] + [file_name_parts1[-1]])
        img3 = Image.open(f'F:/avm31-60/nii/png/{name}_3/{file_name_3}').convert("RGB")
        img3 = np.array(img3)
        # np.resize(img2,(512,512,1))
        # plt.imshow(img2)
        # print(img3.shape)
        combined_image = np.zeros((512, 512, 3), dtype=np.uint8)
        # image = np.concatenate([img1, img2, img3], axis=2)
        # image = np.transpose(image,(2,0,1)) 
        # print(image.shape)
        combined_image[:, :, 0] = img1[:, :, 0] 
        combined_image[:, :, 1] = img2[:, :, 1]  
        combined_image[:, :, 2] = img3[:, :, 2]  

  
        combined_image_pil = Image.fromarray(combined_image)
        combined_image_pil.save(os.path.join(save_path,file_name))

 
