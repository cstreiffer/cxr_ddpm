# Create the data loader
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import cv2
from time import time
import skimage.exposure as hist
from os import makedirs
from os.path import exists, join
from datetime import datetime
from torchvision import transforms
from PIL import Image
import os
import sys

# sys.path.insert(0, "../model/")
from sklearn.preprocessing import OneHotEncoder

class CXRDiffusionDataset(Dataset):

    def __init__(
        self,
        metadata_df_path,
        label_cols,
        downsample_size=224,
        age_stats=None,
        cont_feat_labels=[],
        norm_cont_feat=False,
        cont_feat_stats=None,
        train=False,
    ):
        self.metadata = pd.read_csv(metadata_df_path)
        self.label_cols = label_cols
        self.train = train
        self.age_stats = age_stats
        self.cont_feat_stats = cont_feat_stats
        self.downsample_size = downsample_size

        # Create the transform
        self.transform = transforms.Compose([
            transforms.Resize((downsample_size, downsample_size)),
            transforms.ToTensor(),
        ])

        if age_stats is None:
            self.age_mean = self.metadata["age"].mean()
            self.age_var = self.metadata["age"].var()
            self.metadata["age"] = (
                self.metadata["age"] - self.metadata["age"].mean()
            ) / np.sqrt(self.metadata["age"].var())
        else:
            self.metadata["age"] = (self.metadata["age"] - age_stats[0]) / np.sqrt(
                age_stats[1]
            )

        # Normalize the feats
        if norm_cont_feat:
            # Use them to update
            if cont_feat_stats is None:
                cont_feat_stats = {}
                for feat in cont_feat_labels:
                    mean = self.metadata[feat].mean()
                    var = self.metadata[feat].var()
                    self.metadata[feat] = (self.metadata[feat] - mean) / np.sqrt(var)
                    cont_feat_stats[feat] = [mean, var]
                self.cont_feat_stats = cont_feat_stats
            else:
                for feat in cont_feat_labels:
                    self.metadata[feat] = (self.metadata[feat] - cont_feat_stats[feat][0]) / np.sqrt(
                        cont_feat_stats[feat][1]
                    )

    def __len__(self):
        return len(set(self.metadata.index))

    def __getitem__(self, item):

        image_metadata = self.metadata.loc[item]
        path = image_metadata["file_path"]

        try:
            image = Image.open(path)
        except:
            print("Error in reading file {}".format(path))
            return None

        # Apply the transform
        image = self.transform(image)

        # Load the context
        context = torch.from_numpy(image_metadata[self.label_cols].values.astype(np.float32))

        # Now return
        return image, context


class CXRInferenceDataset(Dataset):

    def __init__(
        self,
        root_path,
        metadata_df_path,
        label_cols,
        cont_label_cols=[],
        multi_view=False,
        include_demo=False,
        downsample_size=224,
        adaptive_norm=True,
        equalize_norm=False,
        brighten=False,
        gaussian_noise=True,
        gaussian_noise_settings=(0, 0.05),
        center_cropped=False,
        center_cropped_size=1400,
        multi_resolution=False,
        multi_resolution_sizes=["full", 1400, 900],
        rand_affine=False,
        age_stats=None,
        cont_label_stats=None,
        enc=None,
        train=False,
        handle_nans="replace_zero",
        handle_minus_one="replace_zero",
        filter_by_first_cxr=False,
        filter_by_cxr_before_echo=False,
        optuna_trial=None,
    ):

        self.metadata = pd.read_csv(metadata_df_path)
        self.root_path = root_path
        self.label_cols = label_cols
        self.cont_label_cols = cont_label_cols

        # Unique deid for each row
        # self.metadata["IMAGE_ID"] = np.unique(
        #     self.metadata["cxr_filename"], return_inverse=True
        # )[-1]

        # self.mapping = self.metadata[["IMAGE_ID", "cxr_filename"]]

        # self.metadata.set_index("IMAGE_ID", inplace=True)
        self.train = train
        self.multi_view = multi_view
        self.include_demo = include_demo
        self.downsample_size = downsample_size
        self.adaptive_norm = adaptive_norm
        self.equalize_norm = equalize_norm
        self.brighten = brighten
        self.gaussian_noise = gaussian_noise
        self.gaussian_noise_settings = gaussian_noise_settings
        self.center_cropped = center_cropped
        self.center_cropped_size = center_cropped_size
        self.multi_resolution = multi_resolution
        self.multi_resolution_sizes = multi_resolution_sizes
        self.rand_affine = rand_affine
        self.age_stats = age_stats
        self.enc = enc

        if self.include_demo:
            demo_vars = self.metadata[["sex", "age"]].to_numpy()

            if enc is None:
                enc = OneHotEncoder()
                self.encoded_sex = enc.fit_transform(
                    demo_vars[:, 0].reshape(-1, 1)
                ).toarray()
                self.enc = enc
            else:
                self.enc = enc
                self.encoded_sex = self.enc.transform(
                    demo_vars[:, 0].reshape(-1, 1)
                ).toarray()

            self.locs_to_ilocs = dict(
                zip(self.metadata.index, np.arange(len(self.metadata.index)))
            )

            if age_stats is None:
                self.age_mean = self.metadata["age"].mean()
                self.age_var = self.metadata["age"].var()
                self.metadata["age"] = (
                    self.metadata["age"] - self.metadata["age"].mean()
                ) / np.sqrt(self.metadata["age"].var())
            else:
                self.metadata["age"] = (self.metadata["age"] - age_stats[0]) / np.sqrt(
                    age_stats[1]
                )

            # for label in self.cont_label_cols:
            #     if cont_label_stats is None:
            #         self.metadata[label] = (
            #             self.metadata[label] - self.metadata[label].mean()
            #         ) / np.sqrt(self.metadata[label].var())
            #     else:
            #         self.metadata[label] = (
            #             self.metadata[label] - cont_label_stats[label][0]
            #         ) / np.sqrt(self.cont_label_stats[label][1])

    def __len__(self):
        return len(set(self.metadata.index))

    def __getitem__(self, item):

        image_metadata = self.metadata.loc[item]

        if self.include_demo:
            demo_index = self.locs_to_ilocs[item]
            encoded_sex = self.encoded_sex[demo_index]
            encoded_age = image_metadata["age"]
            demo_tensor = torch.from_numpy(
                np.concatenate((encoded_sex, np.array([encoded_age])))
            )
            demo_tensor = demo_tensor.float()

        path = image_metadata["file_path"]

        try:
            image_array = [cv2.imread(path)[:, :, 0]]
        except:
            print("Error in reading file {}".format(path))
            return None

        # Downsample as needed
        image_array = [
            zoom_2D(im_arr, (self.downsample_size, self.downsample_size))
            for im_arr in image_array
        ]

        if self.train and self.gaussian_noise:
            mu, std = self.gaussian_noise_settings
            image_array = [
                ds
                + np.random.normal(
                    mu, std, size=(self.downsample_size, self.downsample_size)
                )
                for ds in image_array
            ]

        image_array = [torch.from_numpy(ds).float() for ds in image_array]

        if len(image_array) == 1:
            image_array = image_array[0].unsqueeze(0)
            image_array = image_array.repeat(3, 1, 1)
        else:
            image_array = torch.stack(image_array)

        labels = image_metadata[self.label_cols]
        labels = torch.tensor(labels)

        cont_labels = torch.tensor([])
        if len(self.cont_label_cols) > 0:
            cont_labels = image_metadata[self.cont_label_cols]
            cont_labels = torch.tensor(cont_labels)

        if self.include_demo:
            return image_array, demo_tensor, labels, cont_labels
        else:
            return image_array, labels, cont_labels

def collate_cxr(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def brighten(img):
    cols, rows = img.shape
    brightness = np.sum(img) / (255 * cols * rows)
    minimum_brightness = 0.6
    ratio = brightness / minimum_brightness
    if ratio >= 1:
        return img
    else:
        return cv2.convertScaleAbs(img, alpha=1 / ratio, beta=0)


def adaptive_normalization(tensor, dim3d=False):
    """
    Contrast localized adaptive histogram normalization
    :param tensor: ndarray, 2d or 3d
    :param dim3d: 2D or 3d. If 2d use Scikit, if 3D use the MCLAHe implementation
    :return: normalized image
    """
    return hist.equalize_adapthist(tensor)  # , kernel_size=128, nbins=1024)


def adaptive_normalization_param(tensor, dim3d=False):
    """
    Contrast localized adaptive histogram normalization
    :param tensor: ndarray, 2'd or 3d
    :param dim3d: 2D or 3d. If 2d use Scikit, if 3D use the MCLAHe implementation
    :return: normalized image
    """
    return hist.equalize_adapthist(tensor, kernel_size=128, nbins=1024)


def equalize_normalization(tensor, dim3d=False):
    return hist.equalize_hist(tensor, nbins=256)


def zoom_2D(image, new_shape):
    """
    Uses open CV to resize a 2D image
    :param image: The input image, numpy array
    :param new_shape: New shape, tuple or array
    :return: the resized image
    """

    # OpenCV reverses X and Y axes
    return cv2.resize(
        image, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_CUBIC
    )


def convert_to_jpg(img_arr, equalize_before=True):
    min_p, max_p = img_arr.min(), img_arr.max()
    img_arr_jpg = (((img_arr - min_p) / max_p) * 255).astype(np.uint8)
    if equalize_before:
        img_arr_jpg = cv2.equalizeHist(img_arr_jpg)
    return img_arr_jpg