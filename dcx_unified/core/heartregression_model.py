import torch
import torch.nn as nn
import torch.nn.functional as F


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNetRFull(nn.Module):
    def __init__(self, n_channels, n_classes=1, model_parallelism=False, args='', isHeart=True):

        device_count = torch.cuda.device_count()

        #self.cuda0 = torch.device('cuda:0')
        self.cuda0 = torch.device('cpu') ##
        #self.cuda0 = torch.device('cuda:1')
        self.cuda1 = torch.device('cpu') ##
        self.model_parallelism = model_parallelism
        if self.model_parallelism:
            if device_count > 1:
                self.cuda1 = torch.device('cuda:1')
                print('Using Model Parallelism with 2 gpu')
            else:
                print('Can not use model parallelism! Only found 1 GPU device!')
                self.cuda1 = torch.device('cuda:0')
        self.isHeart = isHeart

        super(UNetRFull, self).__init__()

        input_feature_len = len(args.split(sep=',')) - 1
        # if args.use_sagital:
        #     n_channels += 1

        n_filter = [8, 16, 32, 64, 128]  # [32, 64, 128, 256, 512]  # [64, 128, 256, 512, 1024]  #

        self.inc = inconv(n_channels, n_filter[0]).cpu()#.cuda(self.cuda0)
        self.down1 = down(n_filter[0], n_filter[1]).cpu()#.cuda(self.cuda0)
        self.down2 = down(n_filter[1], n_filter[2]).cpu()#.cuda(self.cuda0)
        self.down3 = down(n_filter[2], n_filter[3]).cpu()#.cuda(self.cuda0)
        self.down4 = down(n_filter[3], n_filter[3]).cpu()#.cuda(self.cuda0)
        self.up1 = up(n_filter[4], n_filter[2]).cpu()#.cuda(self.cuda1)
        self.up2 = up(n_filter[3], n_filter[1]).cpu()#.cuda(self.cuda1)
        self.up3 = up(n_filter[2], n_filter[0]).cpu()#.cuda(self.cuda1)
        self.up4 = up(n_filter[1], n_filter[0]).cpu()#.cuda(self.cuda1)
        self.outc = outconv(n_filter[0], 1).cpu()#.cuda(self.cuda1)
        if isHeart:
            linear_layer = torch.nn.Linear(512 * 512 + input_feature_len, 50)
        else:
            linear_layer = torch.nn.Linear(2048 * 2048 + input_feature_len, 50)
        self.net_linear = torch.nn.Sequential(
            # torch.nn.Linear(8388608, 1000),
            # torch.nn.Linear(2097152, 1000),
            # torch.nn.Linear(512*512 + input_feature_len, 200),
            linear_layer,
            torch.nn.LeakyReLU(),
            torch.nn.Linear(50, 10),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(10, n_classes),
        ).cpu()#.cuda(self.cuda1)
        #self.fc = nn.Linear(512, 512)
    def forward(self, x, input_feat):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        if self.model_parallelism:
            x1 = x1.cuda(self.cuda1)
            x2 = x2.cuda(self.cuda1)
            x3 = x3.cuda(self.cuda1)
            x4 = x4.cuda(self.cuda1)
            x5 = x5.cuda(self.cuda1)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        # if self.model_parallelism:
        #     x = x.cuda(self.cuda0)
        if self.isHeart:
            x = x.view(-1, 512 * 512)
        else:
            x = x.view(-1, 2048 * 2048)
        #xs = torch.cat((x, input_feat.cuda(self.cuda1)), 1) ##
        xs = torch.cat((x, input_feat.cpu()), 1)
        reg_output = self.net_linear(xs).cpu()
        return reg_output