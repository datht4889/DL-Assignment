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
import torchvision
from torchvision.models import resnet18

def Conv(in_channels, out_channels, kernel, padding):
  return nn.Sequential(
    nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
    nn.ReLU(inplace=True),
  )

class ResNetUNet(nn.Module):
  def __init__(self, n_classes=3):
    super().__init__()

    self.base_model = resnet18(pretrained=True)
    self.base_layers = list(self.base_model.children())

    self.layer0 = nn.Sequential(*self.base_layers[:3]) 
    self.layer0_1x1 = Conv(64, 64, 1, 0)
    self.layer1 = nn.Sequential(*self.base_layers[3:5]) 
    self.layer1_1x1 = Conv(64, 64, 1, 0)
    self.layer2 = self.base_layers[5]  
    self.layer2_1x1 = Conv(128, 128, 1, 0)
    self.layer3 = self.base_layers[6]  
    self.layer3_1x1 = Conv(256, 256, 1, 0)
    self.layer4 = self.base_layers[7] 
    self.layer4_1x1 = Conv(512, 512, 1, 0)

    self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    self.conv_up3 = Conv(256 + 512, 512, 3, 1)
    self.conv_up2 = Conv(128 + 512, 256, 3, 1)
    self.conv_up1 = Conv(64 + 256, 256, 3, 1)
    self.conv_up0 = Conv(64 + 256, 128, 3, 1)

    self.conv_origin0 = Conv(3, 64, 3, 1)
    self.conv_origin1 = Conv(64, 64, 3, 1)
    self.conv_origin2 = Conv(64 + 128, 64, 3, 1)

    self.ouput = nn.Conv2d(64, n_classes, 1)

  def forward(self, input):
    x_original = self.conv_origin0(input)
    x_original = self.conv_origin1(x_original)

    layer0 = self.layer0(input)
    layer1 = self.layer1(layer0)
    layer2 = self.layer2(layer1)
    layer3 = self.layer3(layer2)
    layer4 = self.layer4(layer3)

    layer4 = self.layer4_1x1(layer4)
    x = self.upsample(layer4)
    layer3 = self.layer3_1x1(layer3)
    x = torch.cat([x, layer3], dim=1)
    x = self.conv_up3(x)

    x = self.upsample(x)
    layer2 = self.layer2_1x1(layer2)
    x = torch.cat([x, layer2], dim=1)
    x = self.conv_up2(x)

    x = self.upsample(x)
    layer1 = self.layer1_1x1(layer1)
    x = torch.cat([x, layer1], dim=1)
    x = self.conv_up1(x)

    x = self.upsample(x)
    layer0 = self.layer0_1x1(layer0)
    x = torch.cat([x, layer0], dim=1)
    x = self.conv_up0(x)

    x = self.upsample(x)
    x = torch.cat([x, x_original], dim=1)
    x = self.conv_origin2(x)

    out = self.ouput(x)

    return out

transform = Compose([Resize((512, 512), interpolation=InterpolationMode.BILINEAR),
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
    path = '/kaggle/input/bkai-igh-neopolyp/test/test/'
    unet_test_dataset = UNetTestDataClass(path, transform)
    test_dataloader = DataLoader(unet_test_dataset, batch_size=4, shuffle=True)
                        
    pretrained_path = 'unet_model.pth'
    checkpoint = torch.load(pretrained_path)

    model = ResNetUNet()

    optimizer = optim.Adam(params=model.parameters(), lr=2e-4)
    optimizer.load_state_dict(checkpoint['optimizer'])
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint['model'].items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    # model.eval()

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
    main()