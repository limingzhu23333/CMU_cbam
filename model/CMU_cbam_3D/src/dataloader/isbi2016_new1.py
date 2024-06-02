import os
import numpy as np
import scipy
import torch
import torch.utils.data as data
import nibabel as nib
import random
import torch.nn.functional as F
from scipy import ndimage

from scipy.interpolate import interpn

# 假设原始图像和掩码的形状


import SimpleITK as sitk
def interpolate_3d(img_path,label_path, num):
    new_num=0.125
    img_data = sitk.ReadImage(img_path, sitk.sitkInt16)
    img_data = sitk.GetArrayFromImage(img_data)
    img_data = ndimage.zoom(img_data, zoom=(new_num,new_num,new_num), order=3)

    label_data = sitk.ReadImage(label_path, sitk.sitkInt16)
    label_data = sitk.GetArrayFromImage(label_data)
    label_data = ndimage.zoom(label_data, zoom=(new_num, new_num, new_num), order=0)
    return img_data,label_data

class My3DDataset(data.Dataset):
    def __init__(self, split, size=128,patch_size = 8, aug=False):
        super(My3DDataset, self).__init__()
        self.split = split
        self.size = size
        self.patch_size = patch_size
        self.aug = aug

        # load images and labels
        self.image_paths = []
        self.label_paths = []

        root_dir = '/home/nxw/桌面/3dsge/single/data/'

        if split == 'train':
            image_files = sorted(os.listdir(root_dir + '/Train/Image/'))
            label_files = sorted(os.listdir(root_dir + '/Train/Label/'))
            self._populate_paths(image_files, label_files, root_dir, 'Train')
        elif split == 'valid':
            image_files = sorted(os.listdir(root_dir + '/Validation/Image/'))
            label_files = sorted(os.listdir(root_dir + '/Validation/Label/'))
            self._populate_paths(image_files, label_files, root_dir, 'Validation')
        else:
            image_files = sorted(os.listdir(root_dir + '/Test/Image/'))
            label_files = sorted(os.listdir(root_dir + '/Test/Label/'))
            self._populate_paths(image_files, label_files, root_dir, 'Test')

        self.num_samples = len(self.image_paths)

    def _populate_paths(self, image_files, label_files, root_dir, split):
        label_dict = {lf.split('.')[0]: lf for lf in label_files}
        for image_file in image_files:
            base_name = image_file.split('_')[0]
            if base_name in label_dict:
                self.image_paths.append(f'{root_dir}/{split}/Image/{image_file}')
                self.label_paths.append(f'{root_dir}/{split}/Label/{label_dict[base_name]}')

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label_path = self.label_paths[index]

        #使用插值进行大小的调整，应该用窗体去处理 ，交给你了，加油!
        image_data,label_data = interpolate_3d(image_path,label_path, 64)
        image_data = np.transpose(image_data, (2, 1, 0))
        label_data = np.transpose(label_data, (2, 1, 0))
        #label为255，后面代码我没看,我这给你处理了，后面你需要用255你注释调就行，
        label_data[label_data==255]=1


        # Add an extra dimension to image_data
        image_data = np.expand_dims(image_data, axis=0)
        label_data = np.expand_dims(label_data, axis=0)
       
        if self.aug and self.split == 'train':
            image_data, label_data = self.augment(image_data, label_data)

        image_data = torch.from_numpy(image_data).float()
        label_data = torch.from_numpy(label_data).float()
        image_data = np.transpose(image_data,(0,3,2,1))
        label_data = np.transpose(label_data,(0,3,2,1))
        image_data = self.pad_data(image_data)
        label_data = self.pad_data(label_data)
        return {
            'image_path': self.image_paths[index],
            'label_path': self.label_paths[index],
            'image': image_data,
            'label': label_data,
            'idx': self.image_paths[index],
        }

    def __len__(self):
        return self.num_samples

    def load_nii_to_tensor(self, file_path):
        nii_data = nib.load(file_path)
        return np.array(nii_data.get_fdata())
        
    def pad_data(self, data):
        print(data.shape)
        # Pad the data so that each dimension is divisible by patch_size
        _,d, h, w = data.shape
        pad_d = (self.patch_size - d % self.patch_size) % self.patch_size
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        padding = (0, pad_w, 0, pad_h, 0, pad_d)
        data = F.pad(data, padding, mode='constant', value=0)
        return data


    def augment(self, image, label):
        # Example of a simple 3D augmentation: random flip
        if random.random() > 0.5:
            image = np.flip(image, axis=0).copy()
            label = np.flip(label, axis=0).copy()
        if random.random() > 0.5:
            image = np.flip(image, axis=1).copy()
            label = np.flip(label, axis=1).copy()
        if random.random() > 0.5:
            image = np.flip(image, axis=2).copy()
            label = np.flip(label, axis=2).copy()
        
        return image, label

if __name__ == '__main__':
    dataset = My3DDataset(split='train', aug=True)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=1, pin_memory=True)
    
    for batch in train_loader:
        images = batch['image']
        labels = batch['label']
        print(images.shape, labels.shape)
