from typing import Any
from layer import Dataset, Context

def build_feature(context: Context, products: Dataset("products")) -> Any:

    df = products.to_spark()
    feature = df.select("posting_id", "label_group", "image_phash")

    return feature.toPandas()
