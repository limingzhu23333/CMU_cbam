import os


imgs_folder = "Test/Image"
masks_folder = "Test/Label"


imgs_files = os.listdir(imgs_folder)
masks_files = os.listdir(masks_folder)


common_files = set(imgs_files) & set(masks_files)


for img_file in imgs_files:
    if img_file not in common_files:
        os.remove(os.path.join(imgs_folder, img_file))


for mask_file in masks_files:
    if mask_file not in common_files:
        os.remove(os.path.join(masks_folder, mask_file))
