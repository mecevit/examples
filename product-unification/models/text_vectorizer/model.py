# Product Unification Project Example
from layer import Featureset, Dataset, Train, Context
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, IDF


def train_model(context: Context, products: Dataset("products")) -> Pipeline:
    col = "title"
    min_dfs = 2
    regex_tokenizer = RegexTokenizer(gaps=False, pattern= '\w+', inputCol=col, outputCol=col+'Token')
    stop_words_remover = StopWordsRemover(inputCol=col+'Token', outputCol=col+'SWRemoved')
    count_vectorizer = CountVectorizer(minDF=min_dfs, inputCol=col+'SWRemoved', outputCol=col+'TF')
    idf = IDF(inputCol=col+'TF', outputCol=col+'_vector')

    pipeline = Pipeline(stages=[regex_tokenizer, stop_words_remover, count_vectorizer, idf])
    df = products.to_spark()
    model = pipeline.fit(df)
    train = context.train()
    train.log_parameter("Min DF", min_dfs)

    return model
