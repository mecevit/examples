from typing import Any
from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType
from layer import Dataset, Model, Context
import numpy as np
import io
import base64
import gzip


def build_feature(context: Context, products: Dataset("products"), model: Model("product_text_vectorizer")) -> Any:

    def compress_base64(image_vector):
        """
        Compress and encode image vector to base64 string
        :param image_vector: image vector to encode
        :return: base64 string of an image vector
        """
        buff = io.BytesIO()
        np.save(buff, image_vector)
        buff.seek(0)
        encoded = base64.b64encode(gzip.compress(buff.read()))
        encoded_str = encoded.decode("utf-8")
        return encoded_str

    compress = udf(lambda vector: compress_base64(vector), StringType())
    df = products.to_spark()
    transformed_df = model.get_train().transform(df)
    array_vector_df = transformed_df.withColumn('compressed_vector', compress('title_vector'))
    feature = array_vector_df.select("posting_id", col("compressed_vector").alias('title_vector'))

    return feature.toPandas()
