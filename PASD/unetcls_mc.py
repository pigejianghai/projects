import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor, einsum
from utils import simplex, one_hot

class SurfaceLoss():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: list[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, dist_maps: Tensor) -> Tensor:
        assert simplex(probs)
        assert not one_hot(dist_maps)

        pc = probs[:, self.idc, ...].type(torch.float32)
        dc = dist_maps[:, self.idc, ...].type(torch.float32)

        multipled = einsum("bkwh,bkwh->bkwh", pc, dc)

        loss = multipled.mean()

        return loss


BoundaryLoss = SurfaceLoss

class CombinedLoss(nn.Module):
    def __init__(self, loss1, loss2, alpha=0.5):
        super().__init__()
        self.loss1 = loss1
        self.loss2 = loss2
        self.alpha = alpha
        # self.fold = fold

    def forward(self, pred, label, maps, mask):
        loss1 = self.loss1(pred, label)
        loss2 = self.loss2(maps, mask)#, None
        
        # if self.fold // 10 == 0:
        return  loss1 + self.alpha * loss2
        # else:
        #     return loss1

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
 
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
 
    def forward(self, x):
        return self.double_conv(x)
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
 
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
 
    def forward(self, x):
        return self.maxpool_conv(x)
class Up(nn.Module):
    """Upscaling then double conv"""
 
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
 
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
 
        self.conv = DoubleConv(in_channels, out_channels)
 
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
 
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
 
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
 
    def forward(self, x):
        return self.conv(x)
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
 
        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        self.down4 = Down(128, 128)

        self.up1 = Up(256, 64, bilinear)
        self.up2 = Up(128, 32, bilinear)
        self.up3 = Up(64, 16, bilinear)
        self.up4 = Up(32, 16, bilinear)
        self.outc = OutConv(16, n_classes)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(266, 4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
 
    def forward(self, x):
        x1 = self.inc(x)
        # x1 = [64, 1, 448, 448]
        x2 = self.down1(x1)
        # x2 = [128, 1, 224, 224]
        x3 = self.down2(x2)
        # x3 = [256, 1, 112, 112]
        x4 = self.down3(x3)
        # x4 = [512, 1, 56, 56]
        x5 = self.down4(x4)
        # x5 = [512, 1, 28, 28]
        
        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)
        logits = self.outc(x9)

        cls_1 = torch.flatten(self.avg_pool(x5), 1)
        cls_2 = torch.flatten(self.avg_pool(x6), 1)
        cls_3 = torch.flatten(self.avg_pool(x7), 1)
        cls_4 = torch.flatten(self.avg_pool(x8), 1)
        cls_5 = torch.flatten(self.avg_pool(x9), 1)
        cls_6 = torch.flatten(self.avg_pool(logits), 1)
        cls = torch.cat((cls_1, cls_2, cls_3, cls_4, cls_5, cls_6), axis=1)
        cls = self.fc(cls)
        # x = self.outc(x)
        return cls, logits
    
if __name__ == '__main__':
    a = torch.randn(1, 10, 448, 448)
    # b = torch.randn(1, 10, 448, 448)
    label = torch.tensor([[0.5, 0.0, 0.0, 0.0]])
    mask = torch.zeros(1, 10, 448, 448)
    
    net = UNet(n_channels=10, n_classes=10)
    loss_1 = nn.CrossEntropyLoss()
    loss_2 = BoundaryLoss()
    criterion = CombinedLoss(loss1=loss_1, loss2=loss_2, alpha=0.9)
    
    output_labels, output_masks = net(a)
    loss = criterion(output_labels, label, output_masks, mask)
    net.zero_grad()
    loss.backward()
    
    # print(net)
    print(output_labels.shape, output_masks.shape)
    print(loss)