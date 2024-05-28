import torch.nn as nn
import torch

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pooled = self.avg_pool(x)
        max_pooled = self.max_pool(x)
        avg_out = self.fc(avg_pooled)
        max_out = self.fc(max_pooled)
        out = self.sigmoid(avg_out + max_out)
        return out.expand_as(x)  

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv2d(x)
        return self.sigmoid(x).expand_as(x)  

class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()
        

    def forward(self, x):
        ca = self.channel_attention(x)  
        x = ca * x  # 
        sa = self.spatial_attention(x)  # 
        x = sa * x  # 
        return x,ca,sa
        #out = self.channel_attention(x) * x
        #out = self.spatial_attention(out) * out
        #return out


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class UNet_CBAM(nn.Module):  # 
    def __init__(self, in_chans=1,num_classes=4):
        super(UNet_CBAM, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=in_chans, ch_out=16)  # 64
        self.Conv2 = conv_block(ch_in=16, ch_out=32)  # 64 128
        self.Conv3 = conv_block(ch_in=32, ch_out=64)  # 128 256
        self.Conv4 = conv_block(ch_in=64, ch_out=128)  # 256 512
        self.Conv5 = conv_block(ch_in=128, ch_out=256)  # 512 1024

        self.cbam1 = CBAM(channel=16)
        self.cbam2 = CBAM(channel=32)
        self.cbam3 = CBAM(channel=64)
        self.cbam4 = CBAM(channel=128)

        self.Up5 = up_conv(ch_in=256, ch_out=128)  # 1024 512
        self.Up_conv5 = conv_block(ch_in=256, ch_out=128)

        self.Up4 = up_conv(ch_in=128, ch_out=64)  # 512 256
        self.Up_conv4 = conv_block(ch_in=128, ch_out=64)

        self.Up3 = up_conv(ch_in=64, ch_out=32)  # 256 128
        self.Up_conv3 = conv_block(ch_in=64, ch_out=32)

        self.Up2 = up_conv(ch_in=32, ch_out=16)  # 128 64
        self.Up_conv2 = conv_block(ch_in=32, ch_out=16)

        self.Conv_1x1 = nn.Conv2d(16, num_classes, kernel_size=1, stride=1, padding=0)  # 64

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)
        x1,ca1,sa1 = self.cbam1(x1)
        print("Output from CBAM1:", x1)
        
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        x2,ca2,sa2 = self.cbam2(x2) 

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x3,ca3,sa3 = self.cbam3(x3) 
        
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        x4,ca4,sa4 = self.cbam4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        #return d1
        return d1, (ca1, sa1)
if __name__ =="__main__":
    import torch
    from thop import profile
    from pytorch_model_summary import summary
    m = UNet_CBAM(in_chans=1,num_classes=4)
    x = torch.rand(1,1,224,224)

    model = m
    print(summary(model, x, show_input=False, show_hierarchical=False))
    flops, params = profile(model, (x,))
    print('GFLOPs: ', flops/1000000000, 'Mparams: ', params/1000000)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters count", pytorch_total_params)
