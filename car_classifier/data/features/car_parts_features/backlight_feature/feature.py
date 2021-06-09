from layer import Dataset, Model
import pandas as pd
import base64
from io import BytesIO
from PIL import Image
import torch
from torchvision.transforms import functional as F


def build_feature(ds: Dataset("carimages"),
                  model: Model("backlight_detector")) -> any:
    df = ds.to_pandas()

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    # put the model in evaluation mode
    model.eval()

    feature_data = []
    for index, row in df.iterrows():
        img = Image.open(BytesIO(base64.b64decode(row.content)))
        img_input = F.to_tensor(img)

        with torch.no_grad():
            losses, prediction = model([img_input.to(device)])

        img_str = None
        if len(prediction) > 0 and prediction[0]['scores'][0] > 0.85:
            box = prediction[0]['boxes'][0]
            x1 = float(box[0])
            y1 = float(box[1])
            box_w = float(box[2]) - x1
            box_h = float(box[3]) - y1
            img1 = img.crop((x1, y1, x1 + box_w, y1 + box_h))

            buffered = BytesIO()
            img1.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue())

        feature_data.append([row.id, img_str])

    feature_df = pd.DataFrame(feature_data, columns=['id', 'backlight_image'])
    return feature_df
