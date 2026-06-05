import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(scale_factor),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, scale_factor=2):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, channels_latent=64, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear    

        self.inc = DoubleConv(n_channels, channels_latent)
        self.down1 = Down(channels_latent, 2*channels_latent, scale_factor=2)
        self.down2 = Down(2*channels_latent, 4*channels_latent, scale_factor=(2,1))
        self.down3 = Down(4*channels_latent, 8*channels_latent, scale_factor=(2,1))
        factor = 2 if bilinear else 1
        self.down4 = Down(8*channels_latent, 16*channels_latent // factor, scale_factor=(2,1))
        self.up1 = Up(16*channels_latent, 8*channels_latent // factor, bilinear, scale_factor=(2,1))
        self.up2 = Up(8*channels_latent, 4*channels_latent // factor, bilinear, scale_factor=(2,1))
        self.up3 = Up(4*channels_latent, 2*channels_latent // factor, bilinear, scale_factor=(2,1))
        self.up4 = Up(2*channels_latent, channels_latent, bilinear, scale_factor=2)
        self.outc = OutConv(channels_latent, n_classes)

    def forward(self, x):        
        x1 = self.inc(x)        
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)        
        out = self.outc(x)

        # T -> 0-7
        # vz -> 7-14
        # tau -> 14-21
        # logP -> 21-28
        # np.sign(Bx**2-By**2)*np.sqrt(np.abs(Bx**2-By**2)) -> 28-35
        # np.sign(Bx*By)*np.sqrt(np.abs(Bx*By)) -> 35-42
        # Bz -> 42-49
        
        # out[:, 0:7, :, :] += x[:, 0:1, :, :]
        # out[:, 14:28, :, :] += x[:, 0:1, :, :]

        return out