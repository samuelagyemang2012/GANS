import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

PATCH_SIZE = 256
IMAGE_HEIGHT = 400  # 1280 originally
IMAGE_WIDTH = 400  # 1918 originally
HIGH_RES = 128
LOW_RES = HIGH_RES // 4
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN = "checkpoints/gen.pth"
CHECKPOINT_DISC = "checkpoints/disc.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
LR_STEP_SIZE = 2000
LR_GAMMA = 0.5
NUM_EPOCHS = 10000
POOL_SIZE = 50
BATCH_SIZE = 1
LAMBDA_GP = 10
NUM_WORKERS = 4
IMG_CHANNELS = 3
TRAIN_ROOT_DIR = "C:/Users/Administrator/Desktop/datasets/snow100k/training_data_large/test2/clear/"
TRAIN_DEG_DIR = "C:/Users/Administrator/Desktop/datasets/snow100k/training_data/train/deg/"
TRAIN_CLEAR_DIR = "C:/Users/Administrator/Desktop/datasets/snow100k/training_data/train/clear/"
VAL_DEG_DIR = "C:/Users/Administrator/Desktop/datasets/snow100k/training_data/val/deg/"
VAL_CLEAR_DIR = "C:/Users/Administrator/Desktop/datasets/snow100k/training_data/val/clear/"
TOLERANCE = 30

highres_transform = A.Compose(
    [
        # A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ]
)

lowres_transform = A.Compose(
    [
        A.Resize(width=LOW_RES, height=LOW_RES, interpolation=Image.BICUBIC),
        # A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
    ]
)

both_transforms = A.Compose(
    [
        A.RandomCrop(width=HIGH_RES, height=HIGH_RES),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ]
)

test_transform = A.Compose(
    [
        # A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
    ]
)
