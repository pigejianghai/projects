import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

def id_label_list(txt_path):
    id_list, label_list = [], []
    f = open(txt_path, 'r')
    for line in f:
        id_list.append(line.split()[0])
        label_list.append(line.split()[1])
    return id_list, label_list

class MyDataset(Dataset):
    def __init__(self, id_list, label_list, flag='test'):
        self.flag = flag
        self.id_list = id_list
        self.label_list = label_list
        self.transform = transforms.Compose([
            # transforms.Resize((110,110)), 
            transforms.RandomVerticalFlip(), 
            transforms.RandomHorizontalFlip(), 
            transforms.RandomRotation(12), 
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2)), 
        ])
        # self.label_list = self.get_label_list(txt)

    def __getitem__(self, index):
        # fn, label = self.imgs[index]
        fn, label = self.id_list[index], self.label_list[index]
        fn = '/data1/jianghai/DECT/dataset/' + fn + '_0.npy'
        array = np.load(fn)
        array = array.transpose(2, 1, 0)
        img = transforms.ToTensor()(array)
        if self.flag == 'train':
            img = self.transform(img)
        # if self.target_transform is not None:
        #     label = self.target_transform(label)
        img = img.type(torch.FloatTensor)
        label = int(label)
        return img, label

    def __len__(self):
        return len(self.id_list)
    
if __name__ == '__main__':
    path = '/data1/jianghai/DECT/txt/binary/train.txt'
    id_list, label_list = id_label_list(path)

    train_data = MyDataset(id_list=id_list, label_list=label_list)
    train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True)
    
    # net = Net()
    # net.eval()

    for img, label in train_loader:
        print(img.shape, label)
        
        # img = img.cuda()
        # label = label.cuda()
        # out = net(img)

        # pred = torch.argmax(out.data, dim=1)
