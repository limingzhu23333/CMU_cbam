import cv2
import matplotlib.pyplot as plt
import glob
import os
# p = [str(i).zfill(3) for i in range(1, 100)]
# for pname in p:
    input_img = glob.glob(f'D:/jpg_seg/jpg_seg/src/predicted_segmentation_images/*.png')
    file_names = [os.path.basename(file) for file in input_img]
    save_path = "D:/dcm0409"
    i=0
    for file in file_names:
        file_name = "_".join(file.split("_")[:1]+[file.split("_")[-1]])
        # print(file_name)
        prefix,number = filename.split('_')
        new_filename=f'{prefix}_1_{number}'
        ct_image = cv2.imread(f'D:/nii/png/{prefix}_1/{new_filename}')
        segmentation_image = cv2.imread(f'D:/jpg_seg/jpg_seg/src/predicted_segmentation_images/{file_name}', cv2.IMREAD_GRAYSCALE)
    # file_names = [os.path.basename(file) for file ]
    # ct_image = cv2.imread('D:/nii/png/{pname}_1/')
    # # segmentation_image = cv2.imread('E:/jpg_seg/label_segmentation_images/096_146.png', cv2.IMREAD_GRAYSCALE)
    # segmentation_image = cv2.imread(f'D:/jpg_seg/jpg_seg/src/predicted_segmentation_images/*.png', cv2.IMREAD_GRAYSCALE)


        overlay_image = ct_image.copy()
        overlay_image[segmentation_image == 255] = [255, 255, 255]  # 

    # plt.figure(figsize=(10, 10))
    # plt.imshow(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    # plt.show() -->
