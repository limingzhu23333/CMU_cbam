# -*- coding: utf-8 -*-

import os
import argparse
import torch
from torch.utils.data import DataLoader
import time
import sys
original_path = sys.path.copy()
sys.path.append('../')#cause

from networks.CMUNeXt import cmunext, cmunext_s, cmunext_l
from networks.unet import UNet
from networks.unet_cbam import UNet_CBAM
from networks.CMU_cbam import CMU_CBAM
from networks.res_unet import ResUnet
from src.dataloader.isbi2016_new import myDataset


sys.path = original_path
import torchvision
#from utils.isbi2016_new import norm01, myDataset

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="CMU_CBAM",
                    choices=["CMUNeXt", "CMUNeXt-S", "CMUNeXt-L","UNet","UNet_CBAM","ResUNet","CMU_CBAM"], help='model')
parser.add_argument('--base_dir', type=str, default="/fs0/home/sz2106159/jpg_seg/jpg_seg/datasets", help='dir')
parser.add_argument('--train_file_dir', type=str, default="busi_train_all", help='dir')
parser.add_argument('--val_file_dir', type=str, default="busi_val_all", help='dir')
parser.add_argument('--base_lr', type=float, default=0.001,
                    help='segmentation network learning rate')
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch_size per gpu')
parser.add_argument('--epochs', type=int, default=300,
                    help='')
parser.add_argument('--weights', type=str, default=r'/fs0/home/sz2106159/jpg_seg/jpg_seg/src/logs/test_loss_1_aug_1/model/latest.pkl',
                    help='')
parser.add_argument('--mode', type=str, default=r'test',
                    help='')# optional['val','test']
parser.add_argument('--time', type=bool, default=False,
                    help='如果计算时间就不保存预测和标签')
args = parser.parse_args()



def iou_score(output, target):
    smooth = 1e-5
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5

    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2 * iou) / (iou + 1)


    return iou, dice


def dice_coef(output, target):
    smooth = 1e-5
    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)

def remove_para(weights):
    from collections import  OrderedDict
    new = OrderedDict()
    for K,V in weights.items():
        name = K[7:]
        new [name] = V
    return new
def get_model(args):
    if args.model == "CMUNeXt":
        model = cmunext()
    elif args.model == "CMUNeXt-S":
        model = cmunext_s()
    elif args.model == "CMUNeXt-L":
        model = cmunext_l()
    elif args.model == "UNet":
        model = UNet(in_chns=3,class_num=1)
    elif args.model == "UNet_CBAM":
        model = UNet_CBAM(in_chans=3, num_classes=1)
    elif args.model == "CMU_CBAM":
        model = CMU_CBAM(in_chans=3, num_classes=1)
    elif args.model == "ResUNet":
        model = ResUnet(in_chans=3, num_classes=1)
    else:
        model = None
        print("model err")
        exit(0)
    try:
        model.load_state_dict(torch.load(args.weights, map_location=torch.device('cpu')))

        #model.load_state_dict(torch.load(args.weights))
    except:
        model.load_state_dict(remove_para(torch.load(args.weights, map_location=torch.device('cpu'))))
    return model.cuda()


def binarize_tensor(tensor):
    """
    将张量二值化，阈值取平均值
    Args:
        tensor: 输入的张量

    Returns:
        二值化后的张量
    """
    threshold = tensor.mean()  # 取张量的平均值作为阈值
    device = tensor.device
    binary_tensor = torch.where(tensor > threshold, torch.tensor(1,device=device), torch.tensor(0,device=device))
    return binary_tensor

def inference(args,fold):
    if args.mode == 'test':#infer test imgs
        dataset = myDataset(split='test', aug=False)
    elif args.mode == 'val':#infer val imgs
        dataset = myDataset(split='valid', aug=False)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=1)  # Assuming this function returns the validation dataloader
    model = get_model(args)      # Assuming this function returns the model
    model.eval()
    num = 0
    iou_total = 0
    dice_total = 0

    save_dir = 'test1_predicted_segmentation_images'
    save_dir_label = 'test1_label_segmentation_images'
    os.makedirs(save_dir, exist_ok=True)  # Create directory to save images
    os.makedirs(save_dir_label, exist_ok=True)  # Create directory to save images
    start = time.time()
    with torch.no_grad():
        for i_batch, sampled_batch in enumerate(test_loader):

            input, target = sampled_batch['image'], sampled_batch['label']
            name = sampled_batch['idx']

            name = str(name)

            name = name.split("/")[-1].replace("']",'')
            print(name)


            input = input.cuda()
            target = target.cuda()

            output = model(input)

            iou, dice = iou_score(output, target)
            num += 1
            iou_total += iou
            dice_total += dice

            output = torch.sigmoid(output)
            output = binarize_tensor(output) * 255.0

            if not args.time:
                for i in range(output.shape[1]):
                    output_image = torchvision.transforms.ToPILImage()(output[i].squeeze().cpu().numpy().astype('uint8'))
                    save_path = os.path.join(save_dir, f'{name}')
                    output_image.save(save_path)
                for i in range(target.shape[1]):
                    target = target * 255.0
                    target_image = torchvision.transforms.ToPILImage()(target[i].squeeze().cpu().numpy().astype('uint8'))
                    save_path_label = os.path.join(save_dir_label, f'{name}')
                    target_image.save(save_path_label)




        print("IoU: ", iou_total / num)
        print("DSC: ", dice_total / num)
        print("Total Inference Time: ", time.time()-start)
        print("Avg Inference Time: ", (time.time() - start)/num)


if __name__ == "__main__":
    inference(args, fold=5)#fold5=======>train val:8 2
