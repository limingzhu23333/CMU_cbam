import os
import random
import shutil

imgs_folder = "imgs"
masks_folder = "masks"
train_imgs_folder = "Train/imgs"
val_masks_folder = "Train/masks"


os.makedirs(train_imgs_folder, exist_ok=True)
os.makedirs(val_masks_folder, exist_ok=True)


imgs_files = os.listdir(imgs_folder)

num_train_files = int(0.8 * len(imgs_files))

train_files = random.sample(imgs_files, num_train_files)
for file in train_files:
    shutil.move(os.path.join(imgs_folder, file), os.path.join(train_imgs_folder, file))

for file in train_files:
    mask_file = file.replace(".jpg", ".png") 
    if os.path.exists(os.path.join(masks_folder, mask_file)):
        shutil.move(os.path.join(masks_folder, mask_file), os.path.join(val_masks_folder, mask_file))

print("done！")
