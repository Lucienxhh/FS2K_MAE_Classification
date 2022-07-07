import sys
import os
import requests

import torch
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
from torchvision.utils import save_image
# check whether run in Colab
if 'google.colab' in sys.modules:
    print('Running in Colab.')
    # !pip3 install timm==0.4.5  # 0.3.2 does not work in Colab
    # !git clone https://github.com/facebookresearch/mae.git
    sys.path.append('./mae')
else:
    sys.path.append('../')
import models_mae

###################################################
###################################################
imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def show_image(image, title='', save=False):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    im = torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int()
    
    plt.imshow(im)   
    plt.title(title, fontsize=16)
    plt.axis('off')
    if save:
        x = image * imagenet_std + imagenet_mean; 
        x = torch.einsum('abc->cab', x)
        print(x.shape)
        save_image(x, 'test.png')
    return

def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model

def run_one_image(img, model):
    x = torch.tensor(img)

    # make it a batch-like
    print(x.shape)
    # 升维 x.shape=(224,224,3) -> (1,224,224,3)
    x = x.unsqueeze(dim=0) 
    print(x.shape)
    # x的第二维->第三维，第三维->第四维，第四维->第二维
    # 即 x.shape=(1,224,224,3) -> (1,3,224,224)
    x = torch.einsum('nawc->ncaw', x)
    print(x.shape)

    # run MAEu
    loss, y, mask = model(x.float(), mask_ratio=0.85)
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    
    x = torch.einsum('nchw->nhwc', x)

    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask

    # make the plt figure larger
    plt.rcParams['figure.figsize'] = [24, 24]

    plt.subplot(1, 4, 1)
    show_image(x[0], "original")

    plt.subplot(1, 4, 2)
    show_image(im_masked[0], "masked", save=True)

    plt.subplot(1, 4, 3)
    show_image(y[0], "reconstruction")

    plt.subplot(1, 4, 4)
    show_image(im_paste[0], "reconstruction + visible")

    plt.show()

###################################################
###################################################
# load an image
# img_url = 'https://user-images.githubusercontent.com/11435359/147738734-196fd92f-9260-48d5-ba7e-bf103d29364d.jpg' # fox, from ILSVRC2012_val_00046145
# img_url = 'https://user-images.githubusercontent.com/11435359/147743081-0428eecf-89e5-4e07-8da5-a30fd73cc0ba.jpg' # cucumber, from ILSVRC2012_val_00047851
img = Image.open("mk.png")
img = img.resize((224, 224))
img = np.array(img) / 255.

img = img[:,:,0:3]
print(img.shape)
# assert img.shape == (224, 224, 3)

# normalize by ImageNet mean and std
img = img - imagenet_mean
img = img / imagenet_std

plt.rcParams['figure.figsize'] = [5, 5]
show_image(torch.tensor(img))

###################################################
###################################################
# This is an MAE model trained with pixels as targets for visualization (ViT-Large, training mask ratio=0.75)

# download checkpoint if not exist
# !wget -nc https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_large.pth

chkpt_dir = '../mae_visualize_vit_large.pth'
model_mae = prepare_model(chkpt_dir, 'mae_vit_large_patch16')
print('Model loaded.')

###################################################
###################################################
# make random mask reproducible (comment out to make it change)
torch.manual_seed(2)
print('MAE with pixel reconstruction:')
run_one_image(img, model_mae)