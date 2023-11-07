import numpy as np
import conf
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image


class MapDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        # Charger et séparer l'image X de l'image Y
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = np.array(Image.open(img_path))
        input_image = image[:, :600, :]
        target_image = image[:, 600:, :]


        # Optional step : Data transformation

        # Do a transformation to both input and target
        augmentations = conf.both_transform(image=input_image, image0=target_image)
        input_image = augmentations["image"]
        target_image = augmentations["image0"]

        # Do transformation to input and target indepently
        input_image = conf.transform_only_input(image=input_image)["image"]
        target_image = conf.transform_only_mask(image=target_image)["image"]

        return input_image, target_image

# Satelit view to MAP
class ViewDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        # Charger et séparer l'image X de l'image Y
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = np.array(Image.open(img_path))
        target_image = image[:, :600, :]
        input_image = image[:, 600:, :]


        # Optional step : Data transformation

        # Do a transformation to both input and target
        augmentations = conf.both_transform(image=input_image, image0=target_image)
        input_image = augmentations["image"]
        target_image = augmentations["image0"]

        # Do transformation to input and target indepently
        input_image = conf.transform_only_input(image=input_image)["image"]
        target_image = conf.transform_only_mask(image=target_image)["image"]

        return input_image, target_image

# FACE to COMIQUE 
# COMIQUE TO FACE
class ComicsDataset(Dataset):
    def __init__(self, root_dir,from_faces=True):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir+"faces/")
        self.from_faces = from_faces

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):

        img_file = self.list_files[index]

        if self.from_faces:
            input_img_path = os.path.join(self.root_dir+"faces", img_file)
            output_img_path = os.path.join(self.root_dir+"comics", img_file)
        else : 
            output_img_path = os.path.join(self.root_dir+"faces", img_file)
            input_img_path = os.path.join(self.root_dir+"comics", img_file)
            

        # Data augmentations specefic to pix_2pix

        input_image = np.array(Image.open(input_img_path))
        target_image = np.array(Image.open(output_img_path))

        augmentations = conf.both_transform(image=input_image, image0=target_image)
        input_image = augmentations["image"]
        target_image = augmentations["image0"]

        input_image = conf.transform_only_input(image=input_image)["image"]
        target_image = conf.transform_only_mask(image=target_image)["image"]

        return input_image, target_image



if __name__ == "__main__":
    dataset = ComicsDataset("data/face2comics/",from_faces=True)
    loader = DataLoader(dataset, batch_size=1)
    print(loader.__len__())
    for x, y in loader:
        print(x.shape)
        save_image(x, "x.png")
        save_image(y, "y.png")
        import sys
        sys.exit()
