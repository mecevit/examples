from layer import Featureset, Model
import pandas as pd
import base64
from io import BytesIO
from PIL import Image
import torch
import torchvision.transforms as transforms


def get_transform():
    # Channel wise mean and standard deviation for normalizing according to
    # ImageNet Statistics
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(means, stds)])
    return train_transforms


def build_feature(fs: Featureset("car_parts_features"),
                  model: Model("car_classifier")) -> any:
    df = fs.to_pandas()

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    # put the model in evaluation mode
    model.eval()

    feature_data = []
    input_transform = get_transform()
    for index, row in df.iterrows():
        img = Image.open(BytesIO(base64.b64decode(row.content)))
        img_tensor = input_transform(img)
        img_input = img_tensor.unsqueeze(0).to(device)

        outputs = model(img_input)
        _, preds = torch.max(outputs, dim=1)

        feature_data.append([row.id, preds[0]])

    feature_df = pd.DataFrame(feature_data, columns=['id', 'predicted_year'])
    return feature_df
