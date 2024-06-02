import os, argparse
import torch.nn.functional as F
import torch.utils.data
#from medpy.metric.binary import dc, jc
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import sys
original_path = sys.path.copy()
sys.path.append('../')#cause

from networks.CMU_cbam_3D_before import CMU_CBAM_3D
from src.dataloader.isbi2016_new3 import My3DDataset
import numpy as np
from torch.cuda.amp import autocast, GradScaler

#torch.cuda.set_per_process_memory_fraction(0.6, device=torch.cuda.current_device())

def get_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='CMU_CBAM_3D')
    parser.add_argument('--gpu', type=str, default='0,1,2,3')
    parser.add_argument('--exp_name', type=str, default='test')
    parser.add_argument('--fold', type=str)
    parser.add_argument('--lr_seg', type=float, default=0.001)  #0.0003
    parser.add_argument('--n_epochs', type=int, default=300)  #100
    parser.add_argument('--bt_size', type=int, default=1)  #36
    parser.add_argument('--seg_loss', type=int, default=1, choices=[0, 1])
    parser.add_argument('--aug', type=int, default=1)
    parser.add_argument('--patience', type=int, default=10)  #50
    parser.add_argument('--accumulation_steps', type=int, default=4)  # 

    #log_dir name
    parser.add_argument('--folder_name', type=str, default='Default_folder')

    parse_config = parser.parse_args()
    print(parse_config)
    return parse_config


def ce_loss(pred, gt):
    pred = torch.clamp(pred, 1e-6, 1 - 1e-6)
    return (-gt * torch.log(pred) - (1 - gt) * torch.log(1 - pred)).mean()


def structure_loss(pred, mask):
    """            TransFuse train loss        """
    """            Without sigmoid             """
    weit = 1 + 5 * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def dice_coefficient(y_true, y_pred):
    smooth = 1e-6
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice


def jaccard_coefficient(y_true, y_pred):
    smooth = 1e-6
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    jaccard = (intersection + smooth) / (union + smooth)
    return jaccard

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
    elif args.model == "CMU_CBAM_3D":
        model = CMU_CBAM_3D(in_chans=3, num_classes=1)
    elif args.model == "ResUNet":
        model = ResUnet(in_chans=3, num_classes=1)
    else:
        model = None
        print("model err")
        exit(0)
    return model.cuda()
scaler = GradScaler()
#-------------------------- train func --------------------------#
def train(epoch):
    model.train()
    iteration = 0
    optimizer.zero_grad()  # 
    for batch_idx, batch_data in enumerate(train_loader):
        data = batch_data['image'].cuda().float()
        label = batch_data['label'].cuda().float()
        with autocast():
            output = model(data)
            loss = structure_loss(output, label) / parse_config.accumulation_steps  # 

        scaler.scale(loss).backward()

        if (batch_idx + 1) % parse_config.accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        if (batch_idx + 1) % 10 == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\t[Loss: {:.4f}]'
                .format(epoch, batch_idx * len(data),
                        len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))
import csv
val_loss_file = open('validation_loss.csv', 'w', newline='')
val_loss_writer = csv.writer(val_loss_file)
val_loss_writer.writerow(['Epoch', 'Loss'])
#-------------------------- eval func --------------------------#
def evaluation(epoch, loader, save_attention=False):
    model.eval()
    total_loss = 0
    dice_value = 0
    iou_value = 0
    numm = 0
    for batch_idx, batch_data in enumerate(loader):
        data = batch_data['image'].squeeze(1).cuda().float()
        label = batch_data['label'].squeeze(1).cuda().float()

        with torch.no_grad():
            output = model(data)
            loss = structure_loss(output, label)
            total_loss += loss.item()
            
        output = output.sigmoid().cpu().numpy() > 0.5
        label = label.cpu().numpy()
        assert (output.shape == label.shape)
        dice_ave = dice_coefficient(output, label)
        iou_ave = jaccard_coefficient(output, label)

            
        
        dice_value += dice_ave
        iou_value += iou_ave
        numm += 1

    dice_average = dice_value / numm
    iou_average = iou_value / numm
    average_loss = total_loss / numm
    writer.add_scalar('val_metrics/val_dice', dice_average, epoch)
    writer.add_scalar('val_metrics/val_iou', iou_average, epoch)
    print("Average dice value of evaluation dataset = ", dice_average)
    print("Average iou value of evaluation dataset = ", iou_average)
    val_loss_writer.writerow([epoch, average_loss])
    val_loss_file.flush()
    torch.cuda.empty_cache()
    return dice_average, iou_average, average_loss


import csv

train_loss_file = open('training_loss.csv', 'w', newline='')
train_loss_writer = csv.writer(train_loss_file)
train_loss_writer.writerow(['Epoch', 'Loss'])  

if __name__ == '__main__':
    #-------------------------- get args --------------------------#
    parse_config = get_cfg()

    #-------------------------- build loggers and savers --------------------------#
    exp_name = parse_config.exp_name + '_loss_' + str(
        parse_config.seg_loss) + '_aug_' + str(
            parse_config.aug)

    os.makedirs('logs1/{}'.format(exp_name), exist_ok=True)
    os.makedirs('logs1/{}/model'.format(exp_name), exist_ok=True)
    writer = SummaryWriter('logs1/{}/log'.format(exp_name))
    save_path = 'logs1/{}/model/best.pkl'.format(exp_name)
    latest_path = 'logs1/{}/model/latest.pkl'.format(exp_name)

    EPOCHS = parse_config.n_epochs
    os.environ['CUDA_VISIBLE_DEVICES'] = parse_config.gpu
    device_ids = list(range(torch.cuda.device_count()))
    
    #-------------------------- build dataloaders --------------------------#

    dataset = My3DDataset(split='train', aug=parse_config.aug)
    dataset2 = My3DDataset(split='valid', aug=False)

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=parse_config.bt_size,
                                               shuffle=True,
                                               num_workers=1,
                                               pin_memory=True,
                                               drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        dataset2,
        batch_size=1,  #parse_config.bt_size
        shuffle=False,  #True
        num_workers=1,
        pin_memory=True,
        drop_last=False)  #True

    #-------------------------- build models --------------------------#
    model = get_model(parse_config)
 
    #torch.cuda.set_device(device_ids[0])
    if len(device_ids) > 1:  # 
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=parse_config.lr_seg)
    #scheduler = CosineAnnealingLR(optimizer, T_max=20)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    torch.cuda.empty_cache()
    criteon = [None, ce_loss][parse_config.seg_loss]

    #-------------------------- start training --------------------------#

    max_dice = 0
    max_iou = 0
    best_ep = 0

    min_loss = 10
    min_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        print(optimizer.state_dict()['param_groups'][0]['lr'])
        start = time.time()
        train(epoch)
        dice, iou, loss = evaluation(epoch, val_loader)
        train_loss_writer.writerow([epoch, loss])
        train_loss_file.flush()  # 

        scheduler.step()

        if loss < min_loss:
            min_epoch = epoch
            min_loss = loss
        else:
            if epoch - min_epoch >= parse_config.patience:
                print('Early stopping!')
                break
        if iou > max_iou:
            max_iou = iou
            best_ep = epoch
            torch.save(model.state_dict(), save_path)
        else:
            if epoch - best_ep >= parse_config.patience:
                print('Early stopping!')
                break
        torch.save(model.state_dict(), latest_path)
        time_elapsed = time.time() - start
        print(
            'Training and evaluating on epoch:{} complete in {:.0f}m {:.0f}s'.
            format(epoch, time_elapsed // 60, time_elapsed % 60))
            
train_loss_file.close()
val_loss_file.close()
