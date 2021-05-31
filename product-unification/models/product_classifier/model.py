# Product Unification Project Example

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql.functions import col, pandas_udf, regexp_extract, split,element_at, when, PandasUDFType
from pyspark.mllib.evaluation import MulticlassMetrics
from layer import Featureset, Dataset, Train
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType


def train_model(train: Train, products: Dataset("products"),
                product_features: Featureset("product_features")) -> Pipeline:

    pf = product_features.to_spark()
    pdf = products.to_spark()

    # We join our features with the source dataset to compute the labels
    text_image_df = pf.join(pdf, on=['image'], how='left').select("posting_id",
                                                                  "label_group",
                                                                  "image_vector",
                                                                  "title_vector",
                                                                  "image_phash")

    # Before cross join, we sort and rename our features
    tidf = text_image_df.sort("posting_id").select("posting_id", "label_group",
                                                   "title_vector", "image_vector",
                                                   "image_phash")
    tidf2 = tidf.selectExpr("posting_id as pid", "label_group as lg",
                            "title_vector as tidf", "image_vector as iv",
                            "image_phash as ip")

    # We do a cross join to create product pairs.
    joined = tidf.join(tidf2, tidf.posting_id < tidf2.pid, "cross")

    labeled = joined.withColumn('label', when(col('label_group') == col('lg'),
                                              1.0).otherwise(0.0)).withColumn(
        'has_same_phash',
        when(col('image_phash') == col('ip'), 1.0).otherwise(0.0))

    to_similarity = udf(lambda v1, v2: similarity(v1, v2), DoubleType())

    final_df = labeled.select(
        to_similarity(col("title_vector"), col("tidf")).alias("title_simi"),
        to_similarity(col("image_vector"), col("iv")).alias("img_simi"),
        "has_same_phash", "label")

    # Split data into test and train
    train_set, test_set = final_df.randomSplit([0.7, 0.3])

    # Build our pipeline: VectorAssembler + DecisionTreeClassifier
    assembler = VectorAssembler(
        inputCols=['title_simi', 'img_simi', 'has_same_phash'],
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
    train.log_metric("AUC Score", auc_score)
    train.log_metric("Confusion Matrix", str(metrics.confusionMatrix().toArray()))

    # Log feature importances
    feature_importances = model.bestModel.stages[-1].featureImportances
    train.log_metric("Title Importance", feature_importances.toArray()[0])
    train.log_metric("Image Importance", feature_importances.toArray()[1])

    return model


def cosine_similarity(v1, v2):
    denom = v1.norm(2) * v2.norm(2)
    if denom == 0.0:
        return -1.0
    else:
        return v1.dot(v2) / float(denom)


def similarity(v1, v2):
    vector_simi = cosine_similarity(v1, v2)
    return float(vector_simi)
