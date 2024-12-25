import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from Net import Net
from MyDataLoader import MyDataset

root = r"E:\\image\\train"

train_data = MyDataset(txt=root + '\\' + 'train.txt', transform=transforms.ToTensor())
test_data = MyDataset(txt=root + '\\' + 'test.txt', transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_data, batch_size=2, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=2)

if __name__ == "__main__":
    print("---------------预测分析---------------")
    print("测试集样本：", test_data.__len__())
    model = Net()
    model.load_state_dict(torch.load("model/net20.pth"))
    model.eval()

    total_correct = 0
    for x, y in test_loader:
        y_hat = model(x)
        y_pre = torch.argmax(y_hat, dim=1)
        total_correct += sum(torch.eq(y_pre, y))

    acc = (float(total_correct) / test_data.__len__())*100
    print("总测试样本数量:", test_data.__len__(),
          " 测试成功率:", "%.3f%%" % acc)
