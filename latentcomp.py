import torch
import numpy as np
from utils import show, renormalize, masking
from utils import util, imutil, pbar, losses, inversions
from networks import networks
from PIL import Image
import os
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from networks.psp import id_loss

os.environ['TORCH_EXTENSIONS_DIR'] = '/tmp/torch_cpp/' # needed for stylegan to run
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def load_nets(gan, data):
    
    # bonus stylegan encoder trained on real images + identity loss
    # nets = networks.define_nets('stylegan', 'ffhq', ckpt_path='pretrained_models/sgan_encoders/ffhq_reals_RGBM/netE_epoch_best.pth')
    
    # stylegan trained on gsamples + identity loss
    nets = networks.define_nets('stylegan', 'ffhq')
    return nets


def latent_train(nets, dataset, epochs, batch_size):
    batch_size = 1
    lambda_mse = 1.0 # mse loss
    lambda_lpips = 1.0 # lpips loss
    lambda_z = 0.0 # set lambda_z to 10.0 to optimize the latent first. (optional)
    lambda_id = 0.1 # identity loss 

    # do optional latent optimization
    if lambda_z > 0.:
        checkpoint_dict, opt_losses = inversions.invert_lbfgs(nets, dataset, num_steps=30)
        opt_ws = checkpoint_dict['current_z'].detach().clone().repeat(1, 1, 1)
        # reenable grad after LBFGS
        torch.set_grad_enabled(True)

    # ----- Nets ----------
    netG = nets.generator.eval()
    netE = nets.encoder.eval()
    util.set_requires_grad(False, netG)
    util.set_requires_grad(True, netE)

    # ----- Losses -----
    mse_loss = torch.nn.MSELoss()
    l1_loss = torch.nn.L1Loss()
    perceptual_loss = losses.LPIPS_Loss().cuda().eval()
    #identity_loss = id_loss.IDLoss().cuda().eval()
    #util.set_requires_grad(False, identity_loss)
    util.set_requires_grad(False, perceptual_loss)

    # --- optimizer
    optimizer = torch.optim.Adam(netE.parameters(), lr=0.00005, betas=(0.5, 0.999))

    #target = dataset.repeat(batch_size, 1, 1, 1)

    reshape = torch.nn.AdaptiveAvgPool2d((256, 256)) # lpips and id loss input size

    all_losses = dict(z=[], mse=[], lpips=[], id=[], sim_improvement=[])

    torch.manual_seed(0)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 30-50 steps is about enough    

    for i in pbar(range(epochs)):

        for target in dataloader:

            #print(target.shape)
            
            optimizer.zero_grad()
            
            mask_data = [masking.mask_upsample(target)]
            
            #print("mask_data_size ", mask_data[0][0].shape)
            
            hints = torch.cat([m[0] for m in mask_data])
            masks = torch.cat([m[1] for m in mask_data])

            #print("hints shape ", hints.shape)
            
            encoded = netE(torch.cat([hints, masks], dim=1))
            regenerated = netG(encoded)

            if lambda_z > 0.:
                loss_z = mse_loss(encoded, opt_ws)
            else:
                loss_z = torch.Tensor((0.,)).cuda()
            loss_mse = mse_loss(regenerated, target)
            loss_perceptual = perceptual_loss.forward(reshape(regenerated), reshape(target)).mean() #torch.Tensor((0.,)).cuda() #
            loss_id, sim_improvement, id_logs = torch.Tensor((0.,)).cuda(), torch.Tensor((0.,)).cuda(), torch.Tensor((0.,)).cuda() #identity_loss(reshape(regenerated), reshape(target), reshape(target))
            loss = (lambda_z * loss_z + lambda_mse * loss_mse + lambda_lpips * loss_perceptual + lambda_id * loss_id)
            
            loss = loss_mse
            
            ## loss.backward(retain_graph=True)

            loss.backward()
            optimizer.step()
            
            all_losses['z'].append(loss_z.item())
            all_losses['mse'].append(loss_mse.item())
            all_losses['lpips'].append(loss_perceptual.item())
            all_losses['id'].append(loss_id.item())
            all_losses['sim_improvement'].append(sim_improvement)


    f, ax = plt.subplots(1,4, figsize=(16, 3))
    ax[0].plot(all_losses['z'])
    ax[0].set_title('Z loss')
    ax[1].plot(all_losses['mse'])
    ax[1].set_title('MSE loss')
    ax[2].plot(all_losses['lpips'])
    ax[2].set_title('LPIPS loss')
    ax[3].plot(all_losses['id'])
    ax[3].set_title('ID loss')


def random_composition(nets, img1, img2, outdim, gtype='stylenet'):

    composite = torch.zeros(1, 3, outdim, outdim).cuda()
    mask_composite = torch.zeros_like(composite)[0, [0], :, :]

    with torch.no_grad():
        hints1, mask1 = masking.mask_upsample(img1, mask_cent=0.5 if gtype == 'proggan' else 0.)
        """
        show.a(['Mask', renormalize.as_image(mask1[0]).resize((256, 256), Image.ANTIALIAS)])
        show.a(['Hints', renormalize.as_image((img1 * (1-mask1[:,0]))[0] ).resize((256, 256), Image.ANTIALIAS)])
        show.flush()
        """
        

        hints2, mask2 = masking.mask_upsample(img2, mask_cent=0.5 if gtype == 'proggan' else 0.)
        """
        show.a(['Mask', renormalize.as_image(mask2[0]).resize((256, 256), Image.ANTIALIAS)])
        show.a(['Hints', renormalize.as_image((img2 * (1-mask2[:,0]))[0]).resize((256, 256), Image.ANTIALIAS)])
        show.flush()
        """

        #composite = mask1[:, [0], :, :] + mask2[:, [0], :, :]

        #composite = composite.clamp(1.0, 2.0) -1 

        #comp_image = img1 * (1-mask1[:, 0]) + img2 * (1-composite - (1-mask1[:,0]) )

        #out = nets.invert(comp_image, composite)

        alt_image = img1 * (1-mask1[:, 0]) + img2 * (1- (1-mask1[:,0]) )

        out2 = nets.invert(alt_image, mask1[:, [0], :, :])

        """
        show.a(['Mask Composite', renormalize.as_image(composite[0]).resize((256, 256), Image.ANTIALIAS)])
        show.a(['Image Composite', renormalize.as_image(comp_image[0]).resize((256, 256), Image.ANTIALIAS)])
        show.a(['Image Inverted', renormalize.as_image(out[0]).resize((256, 256), Image.ANTIALIAS)])
        show.flush()
        """

        return out2

        
