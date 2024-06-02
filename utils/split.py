import os
import random
import shutil

# 设置文件夹路径
imgs_folder = "imgs"
masks_folder = "masks"
train_imgs_folder = "Train/imgs"
val_masks_folder = "Train/masks"

# 创建train和val文件夹
os.makedirs(train_imgs_folder, exist_ok=True)
os.makedirs(val_masks_folder, exist_ok=True)

# 获取imgs文件夹内所有文件的列表
imgs_files = os.listdir(imgs_folder)

# 计算80%的文件数量
num_train_files = int(0.8 * len(imgs_files))

# 随机选择80%的文件，并移动到train文件夹的imgs下
train_files = random.sample(imgs_files, num_train_files)
for file in train_files:
    shutil.move(os.path.join(imgs_folder, file), os.path.join(train_imgs_folder, file))

# 对应移动相同文件名的图片到val文件夹的masks下
for file in train_files:
    mask_file = file.replace(".jpg", ".png")  # 假设mask文件的格式为png
    if os.path.exists(os.path.join(masks_folder, mask_file)):
        shutil.move(os.path.join(masks_folder, mask_file), os.path.join(val_masks_folder, mask_file))

print("移动完成！")
