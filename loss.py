from torch import nn, optim
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure, peak_signal_noise_ratio
from torchvision.models import vgg19
import torch


class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vgg = vgg19(pretrained=True).features[:35].eval().to(self.device)

        for param in self.vgg.parameters():
            param.requires_grad = False

        self.loss = nn.MSELoss()

    def forward(self, pred, target):
        vgg_input_features = self.vgg(pred)
        vgg_target_features = self.vgg(target)
        return self.loss(vgg_input_features, vgg_target_features)


class MyLoss(nn.Module):

    def __init__(self):
        super(MyLoss, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vgg19loss = VGGLoss()

    def forward(self, target, pred):
        # MSE
        mse = F.mse_loss(input=pred, target=target)

        # LSSIM
        ssim_loss = 1 - structural_similarity_index_measure(target=target, preds=pred, data_range=1.0)

        # Perceptual Loss
        pl = self.vgg19loss(pred, target)

        # mse + lssim + perceptual loss
        return mse + ssim_loss + (0.3 * pl)


def test():
    pred = torch.randn((1, 3, 400, 400)).cuda()
    gt = torch.randn((1, 3, 400, 400)).cuda()

    loss_net = MyLoss()

    loss = loss_net(gt, pred)
    print(loss)


if __name__ == "__main__":
    test()
