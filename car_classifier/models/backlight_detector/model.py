from typing import Any
from layer import Featureset, Train, Dataset
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
import torch.utils.data
from .cars_dataset import CarsDataset
from . import utils
from . import transforms as T
from .engine import train_one_epoch, evaluate


def get_transform(train):
    transforms = [T.ToTensor()]
    if train:
        # Randomly flip the training images for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def get_instance_segmentation_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.roi_heads.mask_predictor = None
    return model


def train_model(train: Train, ds: Dataset("labeledcars")) -> Any:
    dataset = CarsDataset(ds, get_transform(train=True))
    dataset_test = CarsDataset(ds, get_transform(train=False))

    # split the dataset in train and test set
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - backlights and background (other
    # car parts+environment etc)
    num_classes = 2

    model = get_instance_segmentation_model(num_classes)
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    num_epochs = 4
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch,
                        print_freq=1)
        lr_scheduler.step()
        evaluate(model, data_loader_test, device=device)

    scripted_pytorch_model = torch.jit.script(model)
    return scripted_pytorch_model
