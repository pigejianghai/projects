from Net import Net
from MyDataLoader import MyDataset
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn import metrics
import warnings
import os

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
    warnings.filterwarnings('ignore')
    txt_path = '/data1/jianghai/DECT/txt/binary/test.txt'
    f = open(txt_path, 'r')
    id, label = [], []
    for line in f:
        id.append(line.split()[0])
        label.append(line.split()[1])
    f.close()

    checkpoint_path = '/data1/jianghai/DECT/checkpoint/11211224_net/137.pth.tar'
    test_data = MyDataset(id_list=id, label_list=label, flag='test')
    test_loader = DataLoader(test_data, 
                             num_workers=1, pin_memory=True, 
                             batch_size=1)

    net = Net()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # net = torch.nn.DataParallel(net)
    net.to(device)
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint)
    net.eval()

    test_true = []
    test_pred = []

    index = 1
    count = 0
    for image, label in test_loader:
        image = image.to(device)
        label = label.to(device)
        # mask = mask.to(device)
        out = net(image)

        _, pred = torch.max(out, dim=1)
        # print(label, pred)
        label = label.cpu()
        pred = pred.cpu()
        if pred != label:
            print(index)
            count += 1
        test_true.append(np.array(label))
        test_pred.append(np.array(pred))
        index += 1

    print('Accuracy: ', metrics.accuracy_score(test_true, test_pred))
    print(metrics.classification_report(test_true, test_pred))
    print(count)