import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_FOLDER = "data/face2comics/"
DATA_NAME = "face2comic"
TRAIN_DIR = "data/face2comics/"
VAL_DIR = "data/face2comics/validation/"
EVAL_DIR = "evaluation"

LEARNING_RATE = 2e-4
BATCH_SIZE = 20
NUM_WORKERS = 0
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 200
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_DISC = "weights/face.comic.disc.pth.tar"
CHECKPOINT_GEN = "weights/face.comic.gen.pth.tar"

# Image augmentations operations
both_transform = A.Compose(
    [A.Resize(width=256, height=256),], additional_targets={"image0": "image"},
)

transform_only_input = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(p=0.2),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

transform_only_mask = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)
