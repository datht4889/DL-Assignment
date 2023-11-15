from torchsummary import summary
from torchgeometry.losses import one_hot
import os
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import time
import imageio
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import Resize, PILToTensor, ToPILImage, Compose, InterpolationMode
import torchvision.transforms.v2.functional as TF
from collections import OrderedDict
import wandb
import sys

transform = Compose([Resize((800, 1120), interpolation=InterpolationMode.BILINEAR),
                     PILToTensor()])

class UNetTestDataClass(Dataset):
    def __init__(self, images_path, transform):
        super(UNetTestDataClass, self).__init__()
        
        images_list = os.listdir(images_path)
        images_list = [images_path+i for i in images_list]
        
        self.images_list = images_list
        self.transform = transform
        
    def __getitem__(self, index):
        img_path = self.images_list[index]
        data = Image.open(img_path)
        h = data.size[1]
        w = data.size[0]
        data = self.transform(data) / 255        
        return data, img_path, h, w
    
    def __len__(self):
        return len(self.images_list)



def main():
    unet_test_dataset = UNetTestDataClass(path, transform)
    test_dataloader = DataLoader(unet_test_dataset, batch_size=4, shuffle=True)
                        
    pretrained_path = 'unet_model.pth'
    model = torch.load(pretrained_path)

    model.eval()

    if not os.path.isdir("/kaggle/working/predicted_masks"):
        os.mkdir("/kaggle/working/predicted_masks")
    for _, (img, path, H, W) in enumerate(test_dataloader):
        a = path
        b = img
        h = H
        w = W
    
    with torch.no_grad():
        predicted_mask = model(b)
    for i in range(len(a)):
        image_id = a[i].split('/')[-1].split('.')[0]
        filename = image_id + ".png"
        mask2img = Resize((h[i].item(), w[i].item()), interpolation=InterpolationMode.NEAREST)(ToPILImage()(F.one_hot(torch.argmax(predicted_mask[i], 0)).permute(2, 0, 1).float()))
        mask2img.save(os.path.join("/kaggle/working/predicted_masks/", filename))

    def rle_to_string(runs):
        return ' '.join(str(x) for x in runs)

    def rle_encode_one_mask(mask):
        pixels = mask.flatten()
        pixels[pixels > 0] = 255
        use_padding = False
        if pixels[0] or pixels[-1]:
            use_padding = True
            pixel_padded = np.zeros([len(pixels) + 2], dtype=pixels.dtype)
            pixel_padded[1:-1] = pixels
            pixels = pixel_padded
        
        rle = np.where(pixels[1:] != pixels[:-1])[0] + 2
        if use_padding:
            rle = rle - 1
        rle[1::2] = rle[1::2] - rle[:-1:2]
        return rle_to_string(rle)

    def mask2string(dir):
        ## mask --> string
        strings = []
        ids = []
        ws, hs = [[] for i in range(2)]
        for image_id in os.listdir(dir):
            id = image_id.split('.')[0]
            path = os.path.join(dir, image_id)
            print(path)
            img = cv2.imread(path)[:,:,::-1]
            h, w = img.shape[0], img.shape[1]
            for channel in range(2):
                ws.append(w)
                hs.append(h)
                ids.append(f'{id}_{channel}')
                string = rle_encode_one_mask(img[:,:,channel])
                strings.append(string)
        r = {
            'ids': ids,
            'strings': strings,
        }
        return r

        
    MASK_DIR_PATH = '/kaggle/working/predicted_masks' # change this to the path to your output mask folder
    dir = MASK_DIR_PATH
    res = mask2string(dir)
    df = pd.DataFrame(columns=['Id', 'Expected'])
    df['Id'] = res['ids']
    df['Expected'] = res['strings']
    df.to_csv(r'output.csv', index=False)

if __name__ == '__main__':
    # path to test set
    path = sys.argv[1]
    main()