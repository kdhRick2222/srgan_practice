import argparse
import os
from os import listdir
from math import log10

import pandas as pd
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.utils as utils
## import pytorch_ssim
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from model import *
from datasets import *
from utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--val_file', type=str, required=True)
    parser.add_argument('--outputs_dir', type=str, required=True)
    parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8])
    parser.add_argument('--crop_size', type=int, default=88)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=321)
    args = parser.parse_args()

    args.outputs_dir = os.path.join(args.outputs_dir, 'SRGAN_x{}'.format(args.upscale_factor))

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)

    netG = Generator().to(device)
    netD = Discriminator().to(device)
    G_criterion = Generator_Loss().to(device)

    optimizerG = optim.Adam(netG.parameters())
    optimizerD = optim.Adam(netD.parameters())

    train_dataset = TrainDatasetFromFolder(args.train_file, crop_size=args.crop_size, upscale_factor=args.upscale_factor)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  drop_last=True)
    val_dataset = ValDatasetFromFolder(args.val_file, upscale_factor=args.upscale_factor)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=1)

    print('# generator parameters: ', sum(param.numel() for param in netG.parameters()))
    print('# discriminator parameters: ', sum(param.numel() for param in netD.parameters()))

    results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}

    for epoch in range(1, args.num_epochs+1):
        netG.train()
        netD.train()

        train_bar = tqdm(train_dataloader)

        running_results = {'batch_size': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}

        for data, target in train_bar:
            batch_size = data.size(0)
            running_results['batch_size'] += batch_size

            real_img = Variable(target)
            fake_img_1 = Variable(data)

            real_img = real_img.to(device)
            fake_img_1 = fake_img_1.to(device)

            fake_img = netG(fake_img_1)

            ## train Discriminator
            netD.zero_grad()
            real_out = netD(real_img).mean()
            fake_out = netD(fake_img).mean()
            d_loss = 1 - real_out + fake_out
            d_loss.backward(retain_graph=True)
            optimizerD.step()

            ## train Generator
            netG.zero_grad()
            g_loss = G_criterion(fake_out, fake_img, real_img)
            g_loss.backward()

            fake_img = netG(fake_img_1)
            fake_out = netD(fake_img).mean()

            optimizerG.step()

            running_results['g_loss'] += g_loss.item() * batch_size
            running_results['d_loss'] += d_loss.item() * batch_size
            running_results['d_score'] += real_out.item() * batch_size
            running_results['g_score'] += fake_out.item() * batch_size

            train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
                epoch, args.num_epochs, running_results['d_loss'] / running_results['batch_size'],
                running_results['g_loss'] / running_results['batch_size'],
                running_results['d_score'] / running_results['batch_size'],
                running_results['g_score'] / running_results['batch_size']))


        netG.eval()

        with torch.no_grad():
            val_bar = tqdm(val_dataloader)
            val_results ={'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
            val_images = []

            for val_lr, val_hr_restore, val_hr in val_bar:
                batch_size = val_lr.size(0)
                val_results['batch_sizes'] += batch_size
                lr = val_lr.to(device)
                hr = val_hr.to(device)
                sr = netG(lr)

                batch_mse = ((sr - hr) ** 2).data.mean()
                val_results['mse'] += batch_mse * batch_size
                ##batch_ssim = pytorch_ssim.ssim(sr, hr).item()
                ##val_results['ssims'] += batch_ssim * batch_size
                val_results['psnr'] = 10 * log10((hr.max()**2) / (val_results['mse'] / val_results['batch_sizes']))
                ##val_results['ssim'] = val_results['ssims'] / val_results['batch_sizes']
                val_bar.set_description(
                    desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                        val_results['psnr'], val_results['ssim']
                    ))
                val_images.extend(
                    [display_transform()(val_hr_restore.squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)),
                     display_transform()(sr.data.cpu().squeeze(0))]
                )

            if epoch % 10 == 0 and epoch != 0:
                val_images = torch.stack(val_images)
                val_images = torch.chunk(val_images, val_images.size(0) // 15)
                val_save_bar = tqdm(val_images, desc='[saving training results]')
                index = 1
                for image in val_save_bar:
                    image = utils.make_grid(image, nrow=3, padding=5)
                    utils.save_image(image, args.outputs_dir + 'epoch_%d_index_%d.png' 
                    % (epoch, index), padding=5)
                    index += 1

            # save model parameters
            if epoch % 10 == 0 and epoch != 0:
                torch.save(netG.state_dict(), os.path.join(args.outputs_dir, 'netG/epoch_%d_%d.pth' % (args.upscale_factor, epoch)))
                torch.save(netD.state_dict(), os.path.join(args.outputs_dir, 'netD/epoch_%d_%d.pth' % (args.upscale_factor, epoch)))
            # save loss\scores\psnr\ssim
            results['d_loss'].append(running_results['d_loss'] / running_results['batch_size'])
            results['g_loss'].append(running_results['g_loss'] / running_results['batch_size'])
            results['d_score'].append(running_results['d_score'] / running_results['batch_size'])
            results['g_score'].append(running_results['g_score'] / running_results['batch_size'])
            results['psnr'].append(val_results['psnr'])
            results['ssim'].append(val_results['ssim'])

            if epoch % 10 == 0 and epoch != 0:
                out_path = args.outputs_dir
                data_frame = pd.DataFrame(
                    data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Score_D': results['d_score'],
                      'Score_G': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
                index=range(1, epoch + 1))
                data_frame.to_csv(out_path + 'srf_' + str(args.upscale_factor) + '_train_results.csv', index_label='Epoch')






