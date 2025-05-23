import torch
import torch.nn as nn
import torch.nn.functional as F

def random_weights_init(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.xavier_normal_(m.weight.data)
    elif type(m) == nn.BatchNorm2d:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def convBatch(nin, nout, kernel_size=3, stride=1, padding=1, bias=False, layer=nn.Conv3d, dilation=1):
    return nn.Sequential(
        layer(nin, nout, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, dilation=dilation),
        nn.BatchNorm3d(nout),
        nn.PReLU()
    )

def upSampleConv(nin, nout, kernel_size=3, upscale=2, padding=1, bias=False):
    return nn.Sequential(
        # nn.Upsample(scale_factor=upscale), 
        interpolate(mode='nearest', scale_factor=upscale), 
        convBatch(nin, nout, kernel_size=kernel_size, stride=1, padding=padding, bias=bias), 
        convBatch(nout, nout, kernel_size=3, stride=1, padding=1, bias=bias), 
    )

class interpolate(nn.Module):
    def __init__(self, scale_factor, mode='nearest'):
        super().__init__()

        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, cin):
        return F.interpolate(cin, mode=self.mode, scale_factor=self.scale_factor)

class residualConv(nn.Module):
    def __init__(self, nin, nout):
        super(residualConv, self).__init__()
        self.convs = nn.Sequential(
            convBatch(nin, nout),
            nn.Conv3d(nout, nout, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(nout)
        )
        self.res = nn.Sequential()
        if nin != nout:
            self.res = nn.Sequential(
                nn.Conv3d(nin, nout, kernel_size=1, bias=False),
                nn.BatchNorm3d(nout)
            )

    def forward(self, input):
        out = self.convs(input)
        return F.leaky_relu(out + self.res(input), 0.2)
    
class UNet3D(nn.Module):
    def __init__(self, nin: int, nout: int, nG=64):
        super().__init__()

        # from models.unet_3d import (convBatch,
        #                             residualConv,
        #                             upSampleConv)

        self.conv0 = nn.Sequential(convBatch(nin, nG),
                                   convBatch(nG, nG))
        self.conv1 = nn.Sequential(convBatch(nG * 1, nG * 2, stride=2),
                                   convBatch(nG * 2, nG * 2))
        self.conv2 = nn.Sequential(convBatch(nG * 2, nG * 4, stride=2),
                                   convBatch(nG * 4, nG * 4))

        self.bridge = nn.Sequential(convBatch(nG * 4, nG * 8, stride=2),
                                    residualConv(nG * 8, nG * 8),
                                    convBatch(nG * 8, nG * 8))

        self.deconv1 = upSampleConv(nG * 8, nG * 8)
        self.conv5 = nn.Sequential(convBatch(nG * 12, nG * 4),
                                   convBatch(nG * 4, nG * 4))
        self.deconv2 = upSampleConv(nG * 4, nG * 4)
        self.conv6 = nn.Sequential(convBatch(nG * 6, nG * 2),
                                   convBatch(nG * 2, nG * 2))
        self.deconv3 = upSampleConv(nG * 2, nG * 2)
        self.conv7 = nn.Sequential(convBatch(nG * 3, nG * 1),
                                   convBatch(nG * 1, nG * 1))
        self.final = nn.Conv3d(nG, nout, kernel_size=1)

    def forward(self, input):
        input = input.float()
        x0 = self.conv0(input)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)

        bridge = self.bridge(x2)

        y0 = self.deconv1(bridge)
        # print(f"{x0.shape=} {x1.shape=}")
        # print(f"{y0.shape=} {x2.shape=}")
        y1 = self.deconv2(self.conv5(torch.cat((y0, x2), dim=1)))
        y2 = self.deconv3(self.conv6(torch.cat((y1, x1), dim=1)))
        y3 = self.conv7(torch.cat((y2, x0), dim=1))

        return self.final(y3)

    def init_weights(self, *args, **kwargs):
        self.apply(random_weights_init)

if __name__ == '__main__':
    model = UNet3D(nin=1, nout=1, nG=16)
    a = torch.randn(1, 1, 10, 448, 448)
    mask = torch.zeros(1, 1, 10, 448, 448)
    criterion = nn.BCEWithLogitsLoss()
    
    output = model(a)
    loss = criterion(output, mask)
    model.zero_grad()
    loss.backward()
    
    # print(model)
    print(output.shape, mask.shape)
    print(loss)