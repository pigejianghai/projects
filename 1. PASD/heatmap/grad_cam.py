import os
import numpy as np
import torch
import random
import albumentations as A
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def main():
    
    # seed = 2024
    # random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    
    model = models.resnet18()
    fc_inputs = model.fc.in_features
    model.fc = torch.nn.Linear(fc_inputs, 4)
    model.load_state_dict(torch.load('/data1/jianghai/open_source/neoadjuvant/checkpoint/20240716_151642/Fold_1_Epoch_433.pth.tar'))
    target_layers = [model.layer4[-1]]

    # data_transform = A.Compose([
    #     A.Flip(p=0.5), 
    #     A.Rotate(18), 
    # ])

    # Prepare image
    img_path = '/data1/jianghai/open_source/neoadjuvant/data_240712_all/20_channels/benign/MR0430200.npy'
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    # img = Image.open(img_path).convert('RGB')
    arr = np.load(img_path)#, dtype=np.uint8
    # transformed = data_transform(image=arr)
    # img = transformed['image']
    img = transforms.ToTensor()(arr)
    
    # img_tensor = data_transform(img)
    input_tensor = torch.unsqueeze(img, dim=0)

    # Grad CAM
    cam = GradCAM(model=model, target_layers=target_layers)#, use_cuda=True
    # targets = [ClassifierOutputTarget(281)]     # cat
    targets = [ClassifierOutputTarget(0)]  # dog

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(arr.astype(dtype=np.float32)/255., grayscale_cam, use_rgb=False)

    plt.imshow(visualization)
    plt.show()


if __name__ == '__main__':
    main()
