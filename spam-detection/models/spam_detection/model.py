# Spam Detection Project Example

from typing import Any
from layer import Featureset, Train, Model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)


def train_model(train: Train, model: Model("vectorizer"), sf: Featureset("sms_featureset")) -> Any:

    data = sf.to_pandas()
    vectorizer = model.get_train()

    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data["clean_message"],
        data["is_spam"],
        test_size=0.15,
        random_state=0,
        shuffle=True,
        stratify=data["is_spam"],
    )

    # Transform train & test data
    X_train_trans = vectorizer.transform(X_train).toarray()
    X_test_trans = vectorizer.transform(X_test).toarray()

    # Register input and output data of the model
    train.register_input(X_train_trans)
    train.register_output(y_train)

    spam_classifier = MultinomialNB()

    # Train the model with cross validation
    scores = cross_val_score(spam_classifier, X_train_trans, y_train, cv=10, verbose=3, n_jobs=-1)

    spam_classifier.fit(X_train_trans, y_train)
    train.log_metric("mean_scores", scores.mean())

    y_pred = spam_classifier.predict(X_test_trans)
    train.log_metric("accuracy", accuracy_score(y_test, y_pred))
    train.log_metric("f1_score", f1_score(y_test, y_pred))

    return spam_classifier
