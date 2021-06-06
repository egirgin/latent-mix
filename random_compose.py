import pickle
import random

import torch
import torchvision
from utils import masking
from networks import networks
import os
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

os.environ['TORCH_EXTENSIONS_DIR'] = '/tmp/torch_cpp/' # needed for stylegan to run
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def load_nets(gan, data):
    
    # bonus stylegan encoder trained on real images + identity loss
    # nets = networks.define_nets('stylegan', 'ffhq', ckpt_path='pretrained_models/sgan_encoders/ffhq_reals_RGBM/netE_epoch_best.pth')
    
    # stylegan trained on gsamples + identity loss
    nets = networks.define_nets(gan, data)
    return nets



def random_composition(nets, img1, img2, outdim, gtype='stylenet'):

    composite = torch.zeros(1, 3, outdim, outdim).cuda()
    mask_composite = torch.zeros_like(composite)[0, [0], :, :]

    with torch.no_grad():
        hints1, mask1 = masking.mask_upsample(img1, mask_cent=0.5 if gtype == 'proggan' else 0.)

        hints2, mask2 = masking.mask_upsample(img2, mask_cent=0.5 if gtype == 'proggan' else 0.)
 
        #composite = mask1[:, [0], :, :] + mask2[:, [0], :, :]

        #composite = composite.clamp(1.0, 2.0) -1 

        #comp_image = img1 * (1-mask1[:, 0]) + img2 * (1-composite - (1-mask1[:,0]) )

        #out = nets.invert(comp_image, composite)

        alt_image = img1 * (1-mask1[:, 0]) + img2 * (1- (1-mask1[:,0]) )

        out2 = nets.invert(alt_image, mask1[:, [0], :, :])

        return out2


def inv_dataset(church_imgs, livingroom_imgs):
    net = load_nets(gan="stylegan", data="church")

    inv_images = []

    for img in church_imgs:

        with torch.no_grad():
            rec = net.invert(img.unsqueeze(0).to(device), mask=None)
            inv_images.append(rec)

    inv_images_church = torch.cat([item[0].to(device).unsqueeze(0) for item in inv_images], dim=0)
    del net

    print("end inv church")
    ###########################################################

    net = load_nets(gan="proggan", data="livingroom")

    inv_images = []

    for img in livingroom_imgs:
        with torch.no_grad():
            rec = net.invert(img.unsqueeze(0).to(device), mask=None)
            inv_images.append(rec)
            
    inv_images_livingroom = torch.cat([item[0].to(device).unsqueeze(0) for item in inv_images], dim=0)
    del net

    ###############################################################3
    inv_images = torch.cat([inv_images_church, inv_images_livingroom], dim=0)

    return inv_images


def comp_dataset(church_imgs, livingroom_imgs):
    print("Adding Composite samples")
    net = load_nets(gan="stylegan", data="church")

    comp_images = []

    for i in range(len(church_imgs)):
        img = church_imgs[i]
        with torch.no_grad():
            comp = random_composition(net, img.unsqueeze(0).to(device), random.choice(church_imgs).unsqueeze(0).to(device), net.setting["outdim"], gtype="stylenet")
            comp_images.append(comp)

    comp_images_church = torch.cat([item[0].to(device).unsqueeze(0) for item in comp_images], dim=0)

    ###########################################################
    net = load_nets(gan="proggan", data="livingroom")

    comp_images = []

    for i in range(len(livingroom_imgs)):
        img = livingroom_imgs[i]
        with torch.no_grad():
            comp = random_composition(net, img.unsqueeze(0).to(device), random.choice(livingroom_imgs).unsqueeze(0).to(device), net.setting["outdim"], gtype="proggan")
            comp_images.append(comp)


    comp_images_livingroom = torch.cat([item[0].to(device).unsqueeze(0) for item in comp_images], dim=0)

    ###########################################################

    comp_images = torch.cat([comp_images_church, comp_images_livingroom], dim=0)

    return comp_images

def img_inv(nets, img):
    with torch.no_grad():
        rec = nets.invert(img, mask=None)

    return rec