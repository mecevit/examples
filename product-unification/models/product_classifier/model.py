# Product Unification Project Example

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql.functions import col, pandas_udf, regexp_extract, split,element_at, when, PandasUDFType
from pyspark.mllib.evaluation import MulticlassMetrics
from layer import Featureset, Dataset, Context
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType, ArrayType, FloatType
import numpy as np


def train_model(context: Context, product_features: Featureset("product_features"), product_similarity_features: Featureset("product_similarity_features")) -> Pipeline:

    context.spark().conf.set("spark.sql.shuffle.partitions", 25)

    # Load the data.
    pf = product_features.to_spark()
    similarities = product_similarity_features.to_spark()

    # We join our features with the product featureset to compute the labels
    join_labels = pf.selectExpr("posting_id as pid", "label_group", "image_phash")

    label1_similarity = similarities.join(join_labels, similarities.posting_id == join_labels.pid, how="inner").selectExpr(
                                                                  "posting_id",
                                                                  "posting_id_2",
                                                                  "title_similarity",
                                                                  "image_similarity",
                                                                  "label_group as label_group_1",
                                                                  "image_phash as image_phash_1"
                                                                  )

    label2_similarity = label1_similarity.join(join_labels, label1_similarity.posting_id_2 == join_labels.pid, how="inner").selectExpr(
                                                                  "posting_id",
                                                                  "posting_id_2",
                                                                  "title_similarity",
                                                                  "image_similarity",
                                                                  "label_group_1",
                                                                  "image_phash_1",
                                                                  "label_group as label_group_2",
                                                                  "image_phash as image_phash_2"
                                                                  )

    labeled = label2_similarity.withColumn('label', when(col('label_group_1') == col('label_group_2'),
                                              1.0).otherwise(0.0)).withColumn(
        'has_same_phash',
        when(col('image_phash_1') == col('image_phash_2'), 1.0).otherwise(0.0))

    final_df = labeled.select("title_similarity", "image_similarity", "label", "has_same_phash").cache()

    # Split data into test and train
    train_set, test_set = final_df.randomSplit([0.7, 0.3])

    # Build our pipeline: VectorAssembler + DecisionTreeClassifier
    assembler = VectorAssembler(
        inputCols=['title_similarity', 'image_similarity', 'has_same_phash'],
        outputCol='features')
    dt = DecisionTreeClassifier(labelCol='label', featuresCol='features')
    pipeline = Pipeline(stages=[assembler, dt])

    param_grid = ParamGridBuilder() \
        .addGrid(dt.maxDepth, [3, 5]) \
        .build()

    evaluator = BinaryClassificationEvaluator(rawPredictionCol='rawPrediction',
                                              metricName='areaUnderROC')
    crossval = CrossValidator(estimator=pipeline,
                              estimatorParamMaps=param_grid,
                              evaluator=BinaryClassificationEvaluator(),
                              numFolds=3)

    # Train and predict
    model = crossval.fit(train_set)
    prediction = model.transform(test_set)

    # Log metrics
    auc_score = evaluator.evaluate(prediction)
    metrics = MulticlassMetrics(prediction.select('prediction', 'label').rdd.map(tuple))
    context.train().log_metric("AUC Score", auc_score)

    # Log feature importances
    feature_importances = model.bestModel.stages[-1].featureImportances
    context.train().log_metric("Title Importance", feature_importances.toArray()[0])
    context.train().log_metric("Image Importance", feature_importances.toArray()[1])

    return model
