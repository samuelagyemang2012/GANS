import os
import os.path
import torch
import sys
import numpy as np
from torchvision import transforms
from PIL import Image
import config
from torchvision.utils import save_image
from torchmetrics.functional import structural_similarity_index_measure, peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import cv2


def process_tensor(tensor):
    tensor = tensor.detach().squeeze(0).permute(1, 2, 0).cpu().numpy()
    tensor = cv2.cvtColor(tensor, cv2.COLOR_BGR2RGB)
    return tensor


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    # print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    # model.load_state_dict(checkpoint)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def get_SSIM(image1, image2, is_multichannel=True):
    if is_multichannel:
        score, _ = structural_similarity(image1, image2, full=True, multichannel=True)
        return score
    else:
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        score, _ = structural_similarity(image1, image2, full=True)
        return score


def get_psnr(image1, image2, max_value=255):
    mse = np.mean((np.array(image1, dtype=np.float32) - np.array(image2, dtype=np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))


def plot_examples(test_folder_path, dest_path, gen_model, image_dim):
    ssims = []
    psnrs = []
    files = os.listdir(test_folder_path)

    gen_model.eval()
    for file in files:
        image = Image.open(test_folder_path + file)
        to_tensor = transforms.Compose([
            transforms.Resize((image_dim, image_dim)),
            transforms.ToTensor()
        ])
        img_ = to_tensor(image)

        with torch.no_grad():
            res_img = gen_model(img_.unsqueeze(0).to(config.DEVICE))

            img_ = process_tensor(img_)
            res_img = process_tensor(res_img)

            ssim = get_SSIM(img_, res_img, is_multichannel=True)
            psnr = get_psnr(img_, res_img, max_value=1)

            ssims.append(ssim.item())
            psnrs.append(psnr.item())

            cv2.imwrite(dest_path + file, res_img * 255)
            gen_model.train()

    avg_ssim = sum(ssims) / len(ssims)
    avg_psnr = sum(psnrs) / len(psnrs)

    print("avg ssim: {} avg psnr: {}".format(avg_ssim, avg_psnr))
