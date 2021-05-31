from typing import Any
from layer import Dataset, Model


def build_feature(products: Dataset("products"), model: Model("product_text_vectorizer")) -> Any:
    df = products.to_spark()
    transformed_df = model.transform(df)

    return transformed_df.select("posting_id", "title_vector")
