import numpy as np
from PIL import Image as ImPIL
from torchvision import transforms
from torch.utils.data import Dataset
import pandas as PD
from skimage import io
from skimage import segmentation
from skimage import morphology
import warnings
from data_transforms import PairedTransforms
from random import random as rand
warnings.filterwarnings("ignore")


file_list = PD.read_csv("./data/segmentation.csv")

"""
I am only considering the MonuSeg dataset because that is small enough to
compress and transmit
"""

file_list = file_list.loc[file_list["dataset"] == "monuseg"]
file_list.reset_index(drop=True, inplace=True)
# creating traning and validation labels

idx = [(database, np.where(file_list['dataset'] == database)[0]) for database
       in np.unique(list(file_list['dataset'].values))]

for data in idx:
    if len(np.unique(list(file_list['mode'][data[1]].values))) == 1:
        file_list['mode'][data[1][0:int(0.7*len(data[1]))]] = 'train'
        file_list['mode'][data[1][int(0.7*len(data[1])):]] = 'valid'


class nuclei_images(Dataset):
    r""" Expects file contaning image indices.
    train_folder: input folder where traning images reside
    valid_folder: folder where validation images reside
    This dataset is different than the segmentation dataset
    """
    def __init__(self, tensor_size, mode='train'):
        if mode is 'train':
            self.data = file_list[file_list['mode'] == 'train']
            self.data.reset_index(drop=True, inplace=True)
            self.n_ittr = 100
        elif mode is 'valid':
            self.data = file_list[file_list['mode'] == 'valid']
            self.data.reset_index(drop=True, inplace=True)
            self.n_ittr = 1
        elif mode is 'test':
            self.data = file_list[file_list['mode'] == 'test']
            self.data.reset_index(drop=True, inplace=True)
            self.n_ittr = 1
        else:
            raise NotImplementedError("wrong mode of operation")
        self.samples = len(self.data)
        self.to_tensor = transforms.ToTensor()
        self.tensor_size = tensor_size

    def __len__(self):
        return self.n_ittr

    def __getitem__(self, idx):
        image_file = self.data['image'][idx % self.samples]
        mask_file = self.data['mask'][idx % self.samples]
        dataset = self.data['dataset'][idx % self.samples]
        image, mask = self.get_images(image_file, mask_file, dataset)
        return image, mask, dataset

    def get_images(self, image_file, mask_file, dataset):
        image = np.array(io.imread(image_file), dtype='float32')
        mask = np.array(io.imread(mask_file), dtype='float32')

        if np.ndim(image) > 2:
            if image.shape[2] >= 3:
                image = image[:, :, 0:3]
                # image = np.dot(image[..., :3], [0.299, 0.587, 0.114])
        if np.ndim(mask) > 2:
            mask = mask[:, :, 0]

        image = (image-np.min(image))/(np.max(image)-np.min(image))
        mask = (mask-np.min(mask))/(np.max(mask)-np.min(mask))

        if (dataset == 'isbi') or (dataset == 'monuseg'):
            mask = np.array(ImPIL.fromarray(np.flipud(np.array(mask*255,
                                                      dtype='uint8'))
                                            ).rotate(270))
            mask = mask/255.

        # initial set of transformations on image and target data
        PT = PairedTransforms(num_transforms=0.6)
        tf_image, tf_mask = PT(image, mask)

        tf_mask = morphology.label(tf_mask)
        tf_mask = morphology.remove_small_objects(tf_mask, min_size=200)
        boundaries = segmentation.find_boundaries(tf_mask)
        tf_mask = ImPIL.fromarray(np.array(boundaries*255, dtype='uint8'))
        tf_image = ImPIL.fromarray(np.array(tf_image*255, dtype='uint8'))

        # color transformations on image data only
        tv_transforms = transforms.Compose([
                                            transforms.ColorJitter(
                                                brightness=0.2,
                                                contrast=0.3,
                                                saturation=0.1)])

        w, h = tf_image.size
        crop = self._random(w, h, self.tensor_size)
        tf_image = tf_image.crop(crop)
        tf_mask = tf_mask.crop(crop)
        tf_image = tv_transforms(tf_image)
        return self.to_tensor(tf_image), self.to_tensor(tf_mask)

    @staticmethod
    def _random(w, h, t_size):
        crop = [int((h - t_size[2]) * rand()), int((w - t_size[3]) * rand())]
        crop = [crop[1], crop[0], crop[1] + t_size[3], crop[0] + t_size[2]]
        return crop
