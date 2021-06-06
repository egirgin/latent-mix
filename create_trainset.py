import argparse
import pickle
import random
import os

import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from random_compose import inv_dataset, comp_dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
"""
os.environ['TORCH_EXTENSIONS_DIR'] = '/tmp/torch_cpp/' # needed for stylegan to run
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
"""

parser = argparse.ArgumentParser()

parser.add_argument("-a", "--adv", action="store_true")
parser.add_argument("-i", "--inv", action="store_true")
parser.add_argument("-m", "--augmix", action="store_true")
parser.add_argument("-c", "--comp", action="store_true")

args = parser.parse_args()



def gpu_mem():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0) 
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print("Total Mem: {:.2f}".format(t/2**(30)))
    print("Reserved Mem: {:.2f}".format(r/2**(30)))
    print("Allocated Mem: {:.2f}".format(a/2**(30)))
    print("Free Mem: {:.2f}".format(f/2**(30)))



## Load cleanset

transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Resize((256, 256))] )


church = torchvision.datasets.LSUN(root=".", classes=['church_outdoor_val'], transform=transform)
living_room = torchvision.datasets.LSUN(root=".", classes=['living_room_val'], transform=transform)

dataset_default = []
for i in range(len(church)):
    dataset_default.append(church[i])
    dataset_default.append((living_room[i][0], 1))

dataset = dataset_default

church_imgs = [item[0] for item in church] # get imgs

labels_church = torch.zeros(len(church_imgs), 1).to(device)#torch.tensor([item[1] for item in inv_dataset]).to(device)

livingroom_imgs = [item[0] for item in living_room] # get imgs

labels_livingroom = torch.ones(len(livingroom_imgs), 1).to(device)#torch.tensor([item[1] for item in inv_dataset]).to(device)


## ADV SET
if args.adv:
    print("Adding Adv samples")

    with open("./pgd.pickle", "rb") as adv_file:
        adv_samples = pickle.load(adv_file)      
        
        
    with open("./labels.pickle", "rb") as label_file:
        labels = pickle.load(label_file)

    adv_set = [(adv_samples[i], labels[i]) for i in range(len(labels))]

    dataset += adv_set
    print("Dataset size: {}".format(len(dataset)))


if args.inv: # Image inversion
    inv_images = inv_dataset(church_imgs, livingroom_imgs)
    inv_labels = torch.cat([labels_church, labels_livingroom], dim=0)

    dataset += [(inv_images[i], inv_labels[i]) for i in range(len(inv_labels))] 
    print("Dataset size: {}".format(len(dataset)))

    del inv_images, inv_labels

if args.comp:
    comp_images = comp_dataset(church_imgs, livingroom_imgs)
    comp_labels = torch.cat([labels_church, labels_livingroom], dim=0)

    comp_set = [(comp_images[i], comp_labels[i]) for i in range(len(comp_labels))]

    dataset += comp_set
    print("Dataset size: {}".format(len(dataset)))


if args.augmix:
    from augment_and_mix import augment_and_mix

    mixed_imgs = [item[0].permute(1, 2, 0).to("cpu").numpy() for item in dataset_default]
    mixed_labels = torch.tensor([item[1] for item in dataset_default]).to(device)

    mixed_dataset = []

    for img in mixed_imgs:
        mixed_dataset.append(augment_and_mix(img))

    mixed_set = [(torch.from_numpy(item).permute(2,0,1).to(device, dtype=torch.float).unsqueeze(0), mixed_labels[i])  for i, item in enumerate(mixed_dataset)]
    
    dataset += mixed_set
    print("Dataset size: {}".format(len(dataset)))



print("Final size: {}".format(len(dataset)))

