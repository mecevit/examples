import io
import base64
import gzip
from typing import Any, Iterator

import numpy as np
from layer import Context, Dataset, Featureset
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType, ArrayType, FloatType
from pyspark.sql.functions import pandas_udf, unbase64, col


def build_feature(context: Context, product_features: Featureset("product_features")) -> Any:

    """
    Compute similarity of the product-to-product title vector

    :param context: layer context.
    :param product_features.
    """
    context.spark().conf.set("spark.sql.shuffle.partitions", 20)

    def decompress_base64(base64_str):
        """
        Decompress and decode a vector stored as
        a base64-encoded gzip-ed string
        """
        bin_repr = base64_str.encode("utf-8")
        binary = base64.b64decode(bin_repr)
        uncompressed = gzip.decompress(binary)
        buff = io.BytesIO()
        buff.write(uncompressed)
        buff.seek(0)
        arr = np.load(buff, allow_pickle=True, encoding='bytes')
        vector = Vectors.dense(arr)
        return vector

    def cosine_similarity(v1, v2):
        denom = v1.norm(2) * v2.norm(2)
        if denom == 0.0:
            return -1.0
        else:
            return v1.dot(v2) / float(denom)

    def similarity(v1, v2):
        vector_simi = cosine_similarity(v1, v2)
        return float(vector_simi)

    # Load the data
    decompress = udf(lambda vector_str: decompress_base64(vector_str), VectorUDT())
    title_df = product_features.to_spark()
    titles = title_df.select("posting_id", decompress("title_vector").alias("title_vector")).sort("posting_id").repartition(10).cache()

    # Before cross join, we sort and rename our features
    titles_bis = titles.selectExpr("posting_id as posting_id_2", "title_vector as tv")

    # We do a cross join to create product pairs.
    joined = titles.join(titles_bis, titles.posting_id < titles_bis.posting_id_2, "cross")

    to_similarity = udf(lambda v1, v2: similarity(v1, v2), DoubleType())

    cross_df = joined.select("posting_id", "posting_id_2", to_similarity(col("title_vector"), col("tv")).alias("title_similarity"))

    return cross_df.toPandas()
