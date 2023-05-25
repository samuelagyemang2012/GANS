import os
import warnings

warnings.filterwarnings("ignore", category=Warning)
import config
from model import Generator, Discriminator, eff_Discriminator
from data import DegDataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from loss import MyLoss
from tqdm import tqdm
from utils import save_checkpoint, load_checkpoint, plot_examples
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from early_stopping import EarlyStopping
from model_checkpoint import ModelCheckpoint

# parser = argparse.ArgumentParser(description='image-dehazing')
# parser.add_argument('--data_dir', type=str, default='dataset/indoor',
#                     help='dataset directory')
# parser.add_argument('--save_dir', default='results', help='data save directory')
# parser.add_argument('--batch_size', type=int, default=1, help='batch size for training')
# parser.add_argument('--n_threads', type=int, default=8, help='number of threads for data loading')
#
# model
# parser.add_argument('--exp', default='Net1', help='model to select')
#
# optimization
# parser.add_argument('--p_factor', type=float, default=0.5, help='perceptual loss factor')
# parser.add_argument('--g_factor', type=float, default=0.5, help='gan loss factor')
# parser.add_argument('--glr', type=float, default=1e-4, help='generator learning rate')
# parser.add_argument('--dlr', type=float, default=1e-4, help='discriminator learning rate')
# parser.add_argument('--epochs', type=int, default=10000, help='number of epochs to train')
# parser.add_argument('--lr_step_size', type=int, default=2000, help='period of learning rate decay')
# parser.add_argument('--lr_gamma', type=float, default=0.5, help='multiplicative factor of learning rate decay')
# parser.add_argument('--patch_gan', type=int, default=30, help='Path GAN size')
# parser.add_argument('--pool_size', type=int, default=50, help='Buffer size for storing generated samples from G')
#
# misc
# parser.add_argument('--period', type=int, default=1, help='period of printing logs')
# parser.add_argument('--gpu', type=int, required=True, help='gpu index')
#
# args = parser.parse_args()
torch.backends.cudnn.benchmark = True


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)


def train():
    # net
    netG = Generator()
    netG = netG.to(config.DEVICE)

    netD = eff_Discriminator(pretrained=True)
    netD = netD.to(config.DEVICE)

    # loss
    vgg_loss = MyLoss().to(config.DEVICE)

    # opt
    optimizerG = optim.Adam(netG.parameters(), lr=config.LEARNING_RATE)
    optimizerD = optim.Adam(netD.parameters(), lr=config.LEARNING_RATE)

    # lr
    schedulerG = lr_scheduler.MultiStepLR(optimizerG, milestones=[3000, 5000, 8000], gamma=config.LR_GAMMA)
    schedulerD = lr_scheduler.MultiStepLR(optimizerD, milestones=[5000, 7000, 8000], gamma=config.LR_GAMMA)

    # checkpoints
    early_stopping = EarlyStopping(tolerance=30, metric="loss")
    model_checkpoint = ModelCheckpoint(metric="loss")

    dataset = DegDataset(clear_imgs_dir=config.TRAIN_CLEAR_DIR, deg_imgs_dir=config.TRAIN_DEG_DIR)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE,
                            shuffle=True,
                            num_workers=config.NUM_WORKERS)

    if config.LOAD_MODEL:
        print("loading checkpoints")
        load_checkpoint("checkpoints/gen_ad_pl.pth", netG, optimizerG, config.LEARNING_RATE)
        load_checkpoint("checkpoints/disc_ad_pl.pth", netD, optimizerD, config.LEARNING_RATE)

    for epoch in range(config.NUM_EPOCHS):
        print("* Epoch {}/{}".format(epoch + 1, config.NUM_EPOCHS))

        schedulerG.step()
        schedulerD.step()

        netG.train()
        netD.train()

        for batch, (input_image, target_image) in tqdm(enumerate(dataloader)):
            input_image = input_image.to(config.DEVICE)
            target_image = target_image.to(config.DEVICE)

            output_image = netG(input_image)
            netD.zero_grad()

            # real image
            real_output = netD(target_image).mean()
            fake_output = netD(output_image).mean()

            d_loss = 1 - real_output + fake_output
            d_loss.backward(retain_graph=True)
            netG.zero_grad()

            adversarial_loss = torch.mean(1 - fake_output)
            l1 = F.smooth_l1_loss(output_image, target_image)
            p_loss = vgg_loss(output_image, target_image)
            ml = ms_ssim(output_image, target_image, data_range=1.0)

            total_loss = l1 + (0.01 * p_loss) + (0.0005 * adversarial_loss)
            # total_loss = (0.01 * p_loss) + (0.0005 * adversarial_loss)
            total_loss.backward()

            optimizerD.step()
            optimizerG.step()

        print("Train loss: {:.5f}".format(total_loss.item()))

        if config.SAVE_MODEL:
            save_checkpoint(netG, optimizerG, filename="checkpoints/gen.pth")
            save_checkpoint(netD, optimizerD, filename="checkpoints/disc.pth")

        if config.NUM_EPOCHS % 1 == 0:
            plot_examples(test_folder_path="test_images/",
                          dest_path="res/",
                          gen_model=netG,
                          image_dim=config.IMAGE_WIDTH)

        if model_checkpoint(total_loss.item()):
            save_checkpoint(netG, optimizerG, filename="checkpoints/best_gen.pth")
            save_checkpoint(netD, optimizerD, filename="checkpoints/best_disc.pth")
            print("loss improved from {:.4f} to {:.4f}".format(model_checkpoint.get_last_best(), total_loss.item()))

        if early_stopping(total_loss.item()):
            save_checkpoint(netG, optimizerG, filename="checkpoints/last_gen.pth")
            save_checkpoint(netD, optimizerD, filename="checkpoints/last_disc.pth")
            print("Early Stopping on epoch {}".format(epoch + 1))
            break
        print("")


if __name__ == '__main__':
    train()
