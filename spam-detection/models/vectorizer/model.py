# Spam Detection Project Example

from typing import Any
from layer import Featureset, Train
from sklearn.feature_extraction.text import CountVectorizer


def train_model(train: Train, sf: Featureset("sms_featureset")) -> Any:
    data = sf.to_pandas()

    # Transform text data
    vectorizer_model = CountVectorizer(lowercase=False)
    vectorizer_model.fit(data["clean_message"])

    return vectorizer_model
