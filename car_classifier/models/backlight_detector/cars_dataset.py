from PIL import Image
import base64
from io import BytesIO
import torch


class CarsDataset(torch.utils.data.Dataset):
    def __init__(self, ds, transforms=None):
        self.df = ds.to_pandas()
        self.transforms = transforms

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(BytesIO(base64.b64decode(row.image)))

        defs = row.labels.split("\n")

        num_objs = len(defs)
        boxes = []
        classes = []
        for label in defs:
            boxdef = label.split(" ")

            newclass = min(1, int(boxdef[0])) + 1
            classes.append(newclass)

            width = float(boxdef[3]) * 480
            height = float(boxdef[4]) * 360

            xmin = float(boxdef[1]) * 480 - width / 2
            ymin = float(boxdef[2]) * 360 - height / 2
            xmax = xmin + width
            ymax = ymin + height

            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(classes, dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.df.index)
