import os
import argparse
import torch.nn.functional as F
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import sys
import numpy as np
import csv

# Add custom paths
original_path = sys.path.copy()
sys.path.append('../')  # Adjust the path as needed

# Import custom modules
from networks.CMUNeXt import cmunext, cmunext_s, cmunext_l
from networks.unet import UNet
from networks.unet_cbam_1 import UNet_CBAM
from networks.res_unet import ResUnet
from src.dataloader.isbi2016_new import myDataset


def get_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='UNet_CBAM')
    parser.add_argument('--gpu', type=str, default='0,1,2,3,4')
    parser.add_argument('--exp_name', type=str, default='val')
    parser.add_argument('--fold', type=str)
    parser.add_argument('--lr_seg', type=float, default=0.001)
    parser.add_argument('--n_epochs', type=int, default=300)
    parser.add_argument('--bt_size', type=int, default=12)
    parser.add_argument('--seg_loss', type=int, default=1, choices=[0, 1])
    parser.add_argument('--aug', type=int, default=1)
    parser.add_argument('--patience', type=int, default=500)
    parser.add_argument('--folder_name', type=str, default='Default_folder')
    parse_config = parser.parse_args()
    print(parse_config)
    return parse_config


def ce_loss(pred, gt):
    pred = torch.clamp(pred, 1e-6, 1 - 1e-6)
    return (-gt * torch.log(pred) - (1 - gt) * torch.log(1 - pred)).mean()


def structure_loss(pred, mask):
    """ TransFuse train loss """
    """ Without sigmoid """
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
        model = UNet(in_chns=3, class_num=1)
    elif args.model == "UNet_CBAM":
        model = UNet_CBAM(in_chans=3, num_classes=1)
    elif args.model == "ResUNet":
        model = ResUnet(in_chans=3, num_classes=1)
    else:
        model = None
        print("model err")
        exit(0)
    return model.cuda()


def train(epoch, train_loader, model, optimizer):
    model.train()
    for batch_idx, batch_data in enumerate(train_loader):
        data = batch_data['image'].cuda().float()
        label = batch_data['label'].cuda().float()
        output, (ca, sa) = model(data)
        loss = structure_loss(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 10 == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\t[Loss: {:.4f}]'
                .format(epoch, batch_idx * len(data),
                        len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss))


def evaluation(epoch, loader, model, writer, save_attention=False):
    model.eval()
    total_loss = 0
    dice_value = 0
    iou_value = 0
    numm = 0
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(loader):
            data = batch_data['image'].cuda().float()
            label = batch_data['label'].cuda().float()

            output, (ca, sa) = model(data)
            loss = structure_loss(output, label)
            total_loss += loss.item()

            output = output.sigmoid().cpu().numpy() > 0.5
            label = label.cpu().numpy()
            assert (output.shape == label.shape)
            dice_ave = dice_coefficient(output, label)
            iou_ave = jaccard_coefficient(output, label)

            if save_attention and batch_idx == 0:
                torch.save(ca, f'logs1/{exp_name}/channel_attn_epoch_{epoch}.pt')
                torch.save(sa, f'logs1/{exp_name}/spatial_attn_epoch_{epoch}.pt')

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
    return dice_average, iou_average, average_loss


if __name__ == '__main__':
    parse_config = get_cfg()

    exp_name = parse_config.exp_name + '_loss_' + str(parse_config.seg_loss) + '_aug_' + str(parse_config.aug)
    os.makedirs(f'logs1/{exp_name}', exist_ok=True)
    os.makedirs(f'logs1/{exp_name}/model', exist_ok=True)
    writer = SummaryWriter(f'logs1/{exp_name}/log')
    save_path = f'logs1/{exp_name}/model/best.pkl'
    latest_path = f'logs1/{exp_name}/model/latest.pkl'

    EPOCHS = parse_config.n_epochs
    os.environ['CUDA_VISIBLE_DEVICES'] = parse_config.gpu
    device_ids = list(range(torch.cuda.device_count()))

    dataset = myDataset(split='train', aug=parse_config.aug)
    dataset2 = myDataset(split='valid', aug=False)

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=parse_config.bt_size,
                                               shuffle=True,
                                               num_workers=2,
                                               pin_memory=True,
                                               drop_last=True)
    val_loader = torch.utils.data.DataLoader(dataset2,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=2,
                                             pin_memory=True,
                                             drop_last=False)

    model = get_model(parse_config)

    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=parse_config.lr_seg)
    scheduler = CosineAnnealingLR(optimizer, T_max=20)

    train_loss_file = open('training_loss1.csv', 'w', newline='')
    train_loss_writer = csv.writer(train_loss_file)
    train_loss_writer.writerow(['Epoch', 'Loss'])

    val_loss_file = open('validation_loss1.csv', 'w', newline='')
    val_loss_writer = csv.writer(val_loss_file)
    val_loss_writer.writerow(['Epoch', 'Loss'])

    max_dice = 0
    max_iou = 0
    best_ep = 0
    min_loss = float('inf')
    min_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        print(optimizer.state_dict()['param_groups'][0]['lr'])
        start = time.time()
        train(epoch, train_loader, model, optimizer)
        dice, iou, loss = evaluation(epoch, val_loader, model, writer)
        val_loss_writer.writerow([epoch, loss])
        val_loss_file.flush()

        scheduler.step()

        if loss < min_loss:
            min_epoch = epoch
            min_loss = loss
        else:
            if epoch - min_epoch >= parse_config.patience:
                print('Early stopping due to no improvement in loss!')
                break

        if iou > max_iou:
            max_iou = iou
            best_ep = epoch
            torch.save(model.state_dict(), save_path)
        else:
            if epoch - best_ep >= parse_config.patience:
                print('Early stopping due to no improvement in IOU!')
                break

        torch.save(model.state_dict(), latest_path)
        time_elapsed = time.time() - start
        print(f'Training and evaluating on epoch: {epoch} complete in {time_elapsed // 60}m {time_elapsed % 60}s')

    train_loss_file.close()
    val_loss_file.close()
