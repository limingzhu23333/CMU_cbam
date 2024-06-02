# -*- coding: utf-8 -*-
import numpy as np
import torch
from PIL import Image
import os
import sys
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

# 
original_path = sys.path.copy()
sys.path.append('../')

# 
from networks.unet_cbam_before import UNet_CBAM

def load_mri_images(folder_path, transform=None):
    images = []
    for img_name in sorted(os.listdir(folder_path)):
        if img_name.endswith('.png'):
            img_path = os.path.join(folder_path, img_name)
            img = Image.open(img_path).convert('RGB')
            if transform:
                img = transform(img)
            images.append(np.array(img))
    images = np.stack(images, axis=0)  # shape: (num_layers, H, W)
    return torch.tensor(images, dtype=torch.float32)

# 
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

# 
mri_tensor = load_mri_images('./logs/test_loss_1_aug_1/2', transform=transform)
#mri_tensor = mri_tensor.unsqueeze(0)  # 
#mri_tensor = mri_tensor.expand(-1, 3, -1, -1, -1)  # 
print(mri_tensor.shape)

#
model = UNet_CBAM(in_chans=3, num_classes=1)

# 
state_dict = torch.load('./logs/test_loss_1_aug_1/model/best.pkl', map_location=torch.device('cpu'))

# 
new_state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}

# 
model.load_state_dict(new_state_dict)

# 
model.eval()

# 
outputs= model(mri_tensor)

#
print(outputs.shape)
attention = outputs[0,0,:,:]  # 
plt.imshow(attention.detach().numpy(), cmap='hot', interpolation='nearest')
plt.colorbar()
plt.savefig('spatial_attention_map.png')
plt.show()
