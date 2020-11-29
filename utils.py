import torch
import torch.nn as nn
##import torch.backends.cudnn as cudnn
from torchvision.models.vgg import vgg16
from math import log10

# functions about loss
def PSNR(img1, img2):
    return 10. * torch.log10(255. / torch.mean((img1-img2) ** 2))

class Generator_Loss(nn.Module):
    def __init__(self):
        super(Generator_Loss, self).__init__()
        ##cudnn.benchmark = True
        ##device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()

    def forward(self, out_labels, out_images, target_images):
        # adversarial loss
        adversarial_loss = torch.mean(out_labels)
        # perception loss
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        # image loss
        image_loss = self.mse_loss(out_images, target_images)
        tv_loss = self.tv_loss(out_images)
        
        return  0.5 * image_loss + 0.001 * log10(adversarial_loss) + 0.5 * perception_loss + 2e-8 * tv_loss

class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]

if __name__ == "__main__":
    g_loss = Generator_Loss()
    print(g_loss)

