import argparse
import torch.backends.cudnn as cudnn
import os
from tqdm import tqdm

from torch.autograd import Variable
import torchvision.utils as utils
from torch.utils.data.dataloader import DataLoader

from model import Generator
from utils import *
from datasets import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--upscale_factor', default=4, type=int)
    parser.add_argument('--test_image', type=str)
    parser.add_argument('--weights_file', type=str, required=True)
    args = parser.parse_args()

    Upscale_factor = args.upscale_factor
    Test_Image = args.test_image
    weights = args.weights_file

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Generator().to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)
            
    out_path = Test_Image
    #if not os.path.exists(out_path):
        #os.makedirs(join(Test_Image, out_path))
    
    test_set = TestDatasetFromFolder(Test_Image, upscale_factor=Upscale_factor)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
    test_bar = tqdm(test_loader, desc='[testing benachmark datasets]')

    with torch.no_grad():
        for image_name, lr_image, hr_restore_img in test_bar:
            image_name = image_name[0]
            lr_image = Variable(lr_image)
            #hr_image = Variable(hr_image, volatile=True)
            
            lr_image = lr_image.to(device)
            #hr_image = hr_image.to(device)
            
            sr_image = model(lr_image)
            #mse = ((hr_image - sr_image) ** 2).data.mean()
            #psnr = 10 * log10(1 / mse)
            #ssim = pytorch_ssim.ssim(sr_image, hr_image).data[0]
            
            #test_images = torch.stack(
                #[display_transform()(hr_restore_img.squeeze(0)), display_transform()(sr_image.data.cpu().squeeze(0))])
            test_images = torch.stack(
                [(hr_restore_img.squeeze(0)), (sr_image.data.cpu().squeeze(0))])
                
            test_save_bar = tqdm(test_images, desc='[saving training results]')
        
    for image in test_save_bar:

        image = utils.make_grid(test_images, nrow=2, padding=5)
        
        utils.save_image(image, out_path + '_srgan.png', padding=5)
