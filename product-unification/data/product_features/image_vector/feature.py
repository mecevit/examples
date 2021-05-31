import pandas as pd
from PIL import Image
import numpy as np
import io
from typing import Any
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from pyspark.sql.functions import col, pandas_udf, PandasUDFType, udf, unbase64
from pyspark.ml.linalg import Vectors, VectorUDT
from layer import Dataset


def build_feature(products: Dataset("products")) -> Any:
    model = ResNet50(include_top=False)
    bc_model_weights = spark.sparkContext.broadcast(model.get_weights())

    products_df = products.to_spark()
    products_binary_df = products_df.withColumn("content", unbase64("content"))

    to_vector = udf(lambda a: Vectors.dense(a), VectorUDT())
    features_df = products_binary_df.repartition(16).withColumn("image_vector", to_vector(
        featurize_udf("content", bc_model_weights)))

    return features_df.select("posting_id", "image_vector")


def model_fn(bc_model_weights):
    """
    Returns a ResNet50 model with top layer removed and broadcasted pretrained
    weights.
    """
    model = ResNet50(weights=None, include_top=False)
    model.set_weights(bc_model_weights.value)
    return model


def preprocess(content):
    """
    Preprocesses raw image bytes for prediction.
    """
    img = Image.open(io.BytesIO(content)).resize([224, 224])
    arr = img_to_array(img)
    return preprocess_input(arr)


def featurize_series(model, content_series):
    """
    Featurize a pd.Series of raw images using the input model.
    :return: a pd.Series of image features
    """
    input = np.stack(content_series.map(preprocess))
    preds = model.predict(input)
    # For some layers, output features will be multi-dimensional tensors.
    # We flatten the feature tensors to vectors for easier storage in Spark
    # DataFrames.
    output = [p.flatten() for p in preds]
    return pd.Series(output)


@pandas_udf('array<float>', PandasUDFType.SCALAR_ITER)
def featurize_udf(content_series_iter, bc_model_weights):
    """
    This method is a Scalar Iterator pandas UDF wrapping our featurization
    function. The decorator specifies that this returns a Spark DataFrame column
    of type ArrayType(FloatType).

    :param bc_model_weights: Broadcasted pretrained model weights
    :param content_series_iter: This argument is an iterator over batches of
        data, where each batch is a pandas Series of image data.
    """
    # With Scalar Iterator pandas UDFs, we can load the model once and then
    # re-use it for multiple data batches.  This amortizes the overhead of
    # loading big models.
    model = model_fn(bc_model_weights)
    for content_series in content_series_iter:
        yield featurize_series(model, content_series)
