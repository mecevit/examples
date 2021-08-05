import io
import base64
import gzip
from typing import Any, Iterator

import numpy as np
import pandas as pd
from layer import Context, Dataset
from PIL import Image
from pyspark.sql.functions import pandas_udf, unbase64
from pyspark.sql.types import StringType
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array


def build_feature(context: Context, products: Dataset("products")) -> Any:
    """
    Vectorize images using ResNet50 model.

    :param context: layer context.
    :param products: products dataset.
    """
    spark = context.spark().sparkContext
    model_weights = spark.broadcast(ResNet50(include_top=False).get_weights())

    @pandas_udf(returnType=StringType())
    def vectorize_image(
        image_content_iter: Iterator[pd.Series],
    ) -> Iterator[pd.Series]:
        """
        Spark pandas_udf to compute image vectors using ResNet50 model.

        :param image_content_iter: iterator of the image dataset.
        Each iterator item represens a batch of images to process as pandas.Series.
        """
        model = ResNet50(weights=None, include_top=False)
        model.set_weights(model_weights.value)
        for content in image_content_iter:
            _input = np.stack(content.map(preprocess_image_content_resnet50))
            preds = model.predict(_input)
            output = [compress_base64(p.flatten()) for p in preds]
            yield pd.Series(output)

    # distribute the workload evenly, to a number of executors
    products_df = products.to_spark().repartition(5, "image_phash")

    products_binary_df = products_df.withColumn("content", unbase64("content"))
    products_vector_df = products_binary_df.withColumn(
        "image_vector", vectorize_image("content")
    )

    return products_vector_df.select("posting_id", "image_vector").toPandas()


def preprocess_image_content_resnet50(content):
    """
    Prepare raw image data as an input to ResNet50 model.

    :param content: raw image bytes.
    """
    img = Image.open(io.BytesIO(content)).resize([224, 224])
    arr = img_to_array(img)
    return preprocess_input(arr)


def compress_base64(image_vector):
    """
    Compress and encode image vector to base64 string
    :param image_vector: image vector to encode
    :return: base64 string of an image vector
    """
    buff = io.BytesIO()
    np.save(buff, image_vector)
    buff.seek(0)
    return base64.b64encode(gzip.compress(buff.read()))
