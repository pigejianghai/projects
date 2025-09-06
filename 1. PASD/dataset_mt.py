import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.v2 as tv2
import os
import torch.nn.functional as F
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import albumentations as A
# from albumentations.pytorch import ToTensorV2
# from imblearn.over_sampling import RandomOverSampler
from torch.utils.data import Dataset
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_id_label(txt_path : str):
    id_list, label_list = [], []
    f = open(txt_path, 'r')
    for line in f:
        # id_list.append('/data1/jianghai/PA/npy/cropped_448_bc/' + line.split()[0] + '_448_placenta.npy')
        # id_list.append('/data1/jianghai/open_source/neoadjuvant/ViTransformers/multi_task/data/cropped area/ten_placenta+mus/' + line.split()[0] + '.npy')
        # label_list.append(int(line.split()[1]))
        
        id, label = line.split()
        if int(label) == 0:
            id_list.append('/data1/jianghai/open_source/neoadjuvant/data_npy/original_241024/benign/' + id + '.npy')
        else:
            id_list.append('/data1/jianghai/open_source/neoadjuvant/data_npy/original_241024/malign/' + id + '.npy')
        label_list.append(int(label))
    return id_list, label_list

class Mri_Loader(Dataset):
    def __init__(self, path_list : str, label_list : str, flag : str):
        self.path_list, self.label_list = path_list, label_list
        self.flag = flag
        self.transform_train = A.Compose([
            # A.Resize(224, 224), 
            A.Flip(p=0.5), 
            # A.RandomBrightnessContrast(brightness_limit=1, contrast_limit=1, p=0.5), 
            # A.VerticalFlip(p=0.5), 
            A.Rotate(18), 
            # A.ToRGB(), 
        ])
        # self.transform_else = A.Compose([
            # A.Resize(224, 224), 
            # A.Flip(p=0.5), 
            # A.RandomBrightnessContrast(brightness_limit=1, contrast_limit=1, p=0.5), 
            # A.VerticalFlip(p=0.5), 
            # A.Rotate(18), 
            # A.ToRGB(), 
        # ])
    def __getitem__(self, index):
        img_path, label = self.path_list[index], self.label_list[index]
        if 'malign' in img_path:
            msk_path = img_path.replace('malign', 'mask')
        else:
            msk_path = img_path.replace('benign', 'mask')
        img = np.load(img_path).astype(np.float32)
        msk = np.load(msk_path).astype(np.float32)
        
        if self.flag == 'train':
            transformed = self.transform_train(image=img)
            # img = transformed['image']
            # img = transforms.ToTensor()(img)
            transformed = self.transform_train(image=img, mask=msk)
            img, msk = transformed['image'], transformed['mask']
            img, msk = transforms.ToTensor()(img), transforms.ToTensor()(msk)
        else:
            # transformed = self.transform_else(image=img)
            # img = transformed['image']
            img = transforms.ToTensor()(img)
            msk = transforms.ToTensor()(msk)
        # img = img.unsqueeze(0)
        # one_hot = F.one_hot(torch.Tensor(label), num_classes=2)
        return img, msk, int(label)#one_hot#
    def __len__(self):
        return len(self.label_list)
    
if __name__ == "__main__":
    txt_path = "/data1/jianghai/open_source/neoadjuvant/ViTransformers/multi_task/txt/type_multi_class_4.txt"
    id_path, label = get_id_label(txt_path)
    # print(len(id_path), len(label))
    
    img_data = Mri_Loader(id_path, label, flag='train')
    loader = torch.utils.data.DataLoader(dataset=img_data, 
                                         batch_size=1, 
                                         shuffle=False)
    # print(len(img_data))
    # print(img_data.label_list)
    count_0 = 0
    count_1 = 0
    count_2 = 0
    count_3 = 0
    for image, mask, label in loader:#, mask
        if int(label) == 0:
            count_0 += 1
        elif int(label) == 1:
            count_1 += 1
        elif int(label) == 2:
            count_2 += 1
        else:
            count_3 += 1
        # else:
        #     count_2 += 1
        # torch.
        print(image.shape, mask.shape, label)#, mask.shape
        # print(torch.squeeze(mask).shape)
        # size.append(image.shape[2])
        # print(label)
    # print('min: ', min(size))
    print(count_0, count_1, count_2, count_3)    