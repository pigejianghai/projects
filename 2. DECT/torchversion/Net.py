import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #   embedding
        # self.conv = nn.Conv2d(in_channels=11, out_channels=4, kernel_size=1, stride=1, padding=0)
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=11, out_channels=4, 
                      kernel_size=1, stride=1, padding=0), 
            nn.BatchNorm2d(4), 
            nn.ReLU(), 
        )
        # self.embedding = SELayer()
        # 第一个卷积层 输入通道数为11，输出通道数为4，卷积核大小为1
        # self.conv_layer1 = nn.Conv2d(in_channels=11, out_channels=4, kernel_size=1)
        # spectral convolution layer 第二个卷积层 输入通道数为4，输出通道数为4，卷积核大小为3，步长为2，padding为0
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=4, 
                      kernel_size=3, stride=2, padding=0), 
            nn.BatchNorm2d(4), 
            nn.ReLU(), 
        )
        # 第二个卷积层 输入通道数为4，输出通道数为2，卷积核大小为3，步长为2，padding为0
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=2, 
                      kernel_size=3, stride=2, padding=0), 
            nn.BatchNorm2d(2), 
            nn.ReLU(), 
        )
        # 输出大小: 26*26*2 = 1352  输出大小: 2
        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.down = nn.Linear(1352, 512)
        self.fc = nn.Linear(1352, 2)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal(m.weight, mode="fan_out", nonlinearity="relu")
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform(m.weight)
            if isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        #x = x.type(torch.cuda.FloatTensor)
        # x = F.relu(self.conv(x))
        # 110*110*11 -> 110*110*4
        x = self.conv_layer1(x)
        
        # 110*110*4 -> 54*54*4
        x = self.conv_layer2(x)
        # 54*54*4 -> 26*26*2
        x = self.conv_layer3(x)
        # 26*26*2 -> 1352
        x = x.view(x.size(0), -1)
        # 1352 -> 2
        # x = F.softmax(self.linear(x)) 
        # x = F.relu(self.linear(x))
        # x = self.down(x)
        x = torch.sigmoid(self.fc(x))
        return x

if __name__ == '__main__':
    net = Net()
    img = torch.rand([16, 11, 110, 110]).to(device='cuda')
    # m = torch.nn.AdaptiveAvgPool3d((2, 26, 26))
    # output = m(img)
    # print(img.shape, output.shape)
    net.eval()
    pred = net(img)
    print(pred)
    print(net)