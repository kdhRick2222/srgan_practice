import argparse
import os
import torchvision.utils as utils
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from model import Generator
from utils import *
from datasets import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_file', type=str, required=True)
    parser.add_argument('--weights_file', type=str, required=True)
    parser.add_argument('--scale', type=int, default=4)
    args = parser.parse_args()

    out_path = args.image_file

    cudnn.benchmark =  True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = Generator().to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()
    
    #image = Image.open(args.image_file).convert('RGB')
    test_set = ValDatasetFromFolder(args.image_file, upscale_factor=args.scale)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
    val_bar = tqdm(test_loader)
    val_images = []

    for val_lr, val_hr_restore, val_hr in val_bar:
        lr = val_lr.to(device)
        SR_image = model(lr)
        
        val_images = torch.stack([display_transform()(SR_image.data.cpu().squeeze(0))])
        
        #val_images = torch.stack(SR_image)
        #val_images = torch.chunk(val_images, val_images.size(0) // 15)
    #image = utils.make_grid(val_images, nrow=1, padding=0)
    #utils.save_image(image, out_path, padding=0)
    val_save_bar = tqdm(val_images, desc='[saving training results]')
        
    for image in val_save_bar:
        image = utils.make_grid(image, nrow=1, padding=0)
        utils.save_image(image, args.image_file + '_srgan.png', padding=0)
    

