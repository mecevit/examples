from torch.utils.data import Dataset
from PIL import Image
from io import BytesIO
import base64
import torch


class BacklightsDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img = Image.open(BytesIO(base64.b64decode(row.backlight_feature)))
        y_label = torch.tensor(row.label)

        if self.transform is not None:
            img = self.transform(img)

        return img, y_label
