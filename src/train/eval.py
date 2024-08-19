import torch
import torchvision.transforms as transforms
from torchvision.models import inception_v3
import numpy as np
from scipy.stats import entropy
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd

class ISDataset(Dataset):

    def __init__(
        self,
        metadata_df_path,
        downsample_size=224,
        expand_dims=True
    ):
        self.metadata = pd.read_csv(metadata_df_path)
        self.downsample_size = downsample_size
        self.expand_dims = expand_dims

        # Create the transform
        self.transform = transforms.Compose([
            transforms.Resize((downsample_size, downsample_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(set(self.metadata.index))

    def __getitem__(self, item):

        image_metadata = self.metadata.loc[item]
        path = image_metadata["file_path"]

        try:
            image = Image.open(path)
            if self.expand_dims:
              image = image.convert('RGB')
        except:
            print("Error in reading file {}".format(path))
            return None

        # Apply the transform
        image = self.transform(image)

        # Now return
        return image

def inception_score(file_path, cuda=True, splits=1):
    # Create the dataloader
    ds = ISDataset(file_path)
    dataloader = DataLoader(ds, 32, shuffle=False)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False)
    inception_model.eval()
    if cuda:
        inception_model.cuda()

    # Get predictions
    preds = []
    for batch in dataloader:
        with torch.no_grad():
            if cuda:
                batch = batch.cuda()
            pred = inception_model(batch)
            if pred.dim() == 1:
                pred = pred.unsqueeze(0)
            preds.append(pred.cpu().numpy())

    # Now compute the mean kl-div
    preds = np.concatenate(preds, 0)
    scores = []
    for i in range(splits):
        part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
        p_yx = np.exp(part - np.max(part, axis=1, keepdims=True))
        p_yx /= p_yx.sum(axis=1, keepdims=True)
        p_y = np.mean(p_yx, axis=0)
        scores.append(entropy(p_yx, p_y, axis=1).mean())

    return np.exp(np.mean(scores))